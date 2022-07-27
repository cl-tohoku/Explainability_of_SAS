
import pandas as pd
import numpy as np
import argparse
from util import logger as L
from sas import util
from typing import List
import pickle
import json
import torch
from logging import getLogger
from interpretability.explanation import Explanation
from analysis.util import *
from tqdm import tqdm
import time
from collections import defaultdict
from itertools import islice
logger = getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_zero_score", "-izs", dest="include_zero_score", default=False,
                        action='store_true', help="By using this, zero score answer is taken into account in the justification identification")
    parser.add_argument("--train_info_path", "-info")
    parser.add_argument("--print-debug-info", "-debug",
                        dest="debug", default=False, action='store_true', help="")
    parser.add_argument("--flip_mode", dest="flip_mode", default=False,
                        action='store_true', help="削除率の計算方法を「得点が変化するまでの削除率」にする")
    parser.add_argument("-ERASER", action='store_true',
                        default=False, help='use eraser dataset')
    parser.add_argument("--evaluate_size", '-es', type=int,
                        default=1000000000000, help='test data size to evaluate')
    parser.add_argument("--mask_type", '-mt', choices=['zero', 'rand'], default="zero",
                        help="how to mask. choose [zero, rand]. zero replace token 0 vector. rand replace token initalized embedding")
    args = parser.parse_args()
    return args


# def make_masked_data(x, justification_cue,  mask_id, offset=0):
#     # justification_cue[sentence_len:] = -1
#     ret = x.repeat((len(justification_cue), 1))
#     for i, idx in enumerate(torch.argsort(justification_cue, dim=-1, descending=True)):
#         ret[i:, idx+offset] = mask_id
#     return ret


def regularization(target_embeddings):
    _target = torch.clone(target_embeddings)
    _sum = _target.sum(-1).view(-1, 1)
    return _target / _sum


def make_masked_attn(attn, justification_cue, offset=0):
    '''
        attn : torch.tensor(sent_len)
    '''
    ret = attn.repeat((len(attn), 1))
    for i, idx in enumerate(torch.argsort(justification_cue, dim=-1, descending=True)):
        ret[i:, idx + offset] = 0.0
    ret = regularization(ret)
    ret[ret != ret] = 0.0  # nan_to_num
    ret = ret.float()
    # assert 0.999 < ret.sum(-1)[0] and ret.sum(-1)[0] < 1.01
    return ret


def make_faithfulness_eraser_file(info, test_pred_score_list, test_explanation_df, test_x, test_y, test_mask, test_attention, model, mask_id, flip_mode, batch_size=128, include_zero_score=False):
    '''
        eraser法に基づいてfaithfulnessを計算する
        削除率を計算する基準は2通り
        1. 予測得点が0点となるまでの単語の削除率を計算
            - こちらの場合は0点答案は評価対象に含めない
        2. 予測得点が変化するまでの単語の削除率を計算
            - こちらはinclude_zero_scoreの真偽によって0点答案を含めるか含めないか変更する        
    '''

    assert len(test_pred_score_list) == len(
        test_x), f"{len(test_pred_score_list)}, {len(test_x)}"
    output_dict = defaultdict(lambda: defaultdict(float))
    remove_rate_df = pd.DataFrame()

    # attentionはどの時も使用する
    attn = test_explanation_df["Attention_Weights"]

    # 各手法で計算をおこなう
    for method in test_explanation_df.columns:
        # データごと削除率を計算
        remove_rate_list = []
        sentence_len = test_mask.sum(-1)
        if info.BERT:
            # CLS tokenの分を補正　SEPも補正
            sentence_len = sentence_len - 2
        sentence_len = sentence_len.tolist()
        # 0答案の数を数えとく
        zero_cnt = 0
        # 全単語マスクしても0点にならなかった答案の数をカウント(flipしない数もカウント)
        no_terminate_cnt = 0
        model.eval()
        with torch.no_grad():
            bar = tqdm(total=len(sentence_len))
            for batch_x, batch_y, batch_m, batch_len, pred_score, batch_j, batch_attn in zip(test_x, test_y, test_mask, sentence_len, test_pred_score_list, test_explanation_df[method], attn):
                # batch_xとbatch_m のpaddingを消す
                if info.BERT:
                    # [CLS]も消す
                    batch_x = batch_x[1:batch_len + 1]
                    batch_m = batch_m[1:batch_len + 1]
                else:
                    batch_x = batch_x[:batch_len]
                    batch_m = batch_m[:batch_len]
                assert len(batch_j) == len(batch_attn)
                batch_j = torch.tensor(batch_j)
                batch_attn = torch.tensor(batch_attn)
                if not include_zero_score:
                    # 予測が0点の答案は使用しない
                    if pred_score == 0:
                        remove_rate_list.append(0)
                        zero_cnt += 1
                        continue

                # BERTとW2Vで置換するトークンを変える
                # mask_id = info.vocab["[MASK]"] if info.BERT else info.vocab["<unk>"]
                offset = 1 if info.BERT else 0
                # justification cueの値が大きい方から順にmaskしたデータを作成
                assert len(
                    batch_j) == batch_len, f"{len(batch_j)}, {batch_len}\n {batch_j} {batch_x}"
                assert len(batch_attn) == batch_len
                # masked_x = make_masked_data(
                #     batch_x, batch_j, mask_id, offset)

                # マスク部分を0にしたアテンションを作成
                masked_attn = make_masked_attn(batch_attn, batch_j, offset)

                flag = False
                # print(f"original:{pred_score}")
                for batch_index in range(0, batch_len, batch_size):
                    b_masked_attn = masked_attn[batch_index:batch_index +
                                                batch_size].squeeze(0)
                    b_masked_attn = b_masked_attn.to(device)

                    b_batch_x = batch_x.expand(
                        b_masked_attn.size()).to(device)
                    b_batch_m = batch_m.expand(
                        b_batch_x.size()).squeeze(0).to(device)
                    if len(b_batch_x.size()) == 1:
                        b_masked_attn = b_masked_attn.unsqueeze(0)
                        b_batch_x = b_batch_x.unsqueeze(0)
                        b_batch_m = b_batch_m.unsqueeze(0)
                    pred, * \
                        _ = model(ids=b_batch_x, attention_mask=b_batch_m,
                                  justification=b_masked_attn)
                    _, idx = pred.max(-1)
                    # print(f"{idx}")
                    if flip_mode:
                        # 予測が変化するまでの削除率
                        if torch.any(idx != pred_score):
                            masked_cnt = torch.nonzero(idx != pred_score)
                            masked_cnt = masked_cnt[0].to(
                                "cpu").detach().item() + 1
                            # 0点答案が上から何晩目に出てくるか
                            remove_rate = (
                                masked_cnt + batch_index) / batch_len
                            flag = True
                            break

                    else:
                        # if idx[-1] == 0:
                        if torch.any(idx == 0):
                            masked_cnt = torch.nonzero(idx == 0)
                            masked_cnt = masked_cnt[0].to(
                                "cpu").detach().item() + 1
                            # 0点答案が上から何晩目に出てくるか
                            remove_rate = (
                                masked_cnt + batch_index) / batch_len
                            flag = True
                            break
                if flag is False:
                    remove_rate = 1.0
                    no_terminate_cnt += 1
                bar.update(1)
                remove_rate_list.append(remove_rate)

            remove_rate_df[method] = remove_rate_list
            mean_remove_rate = remove_rate_df[method].replace(0, np.NaN).mean()
            logger.info(f"{method} is done")
            logger.info(
                f"zero cnt : {zero_cnt}\tno terminate exist:{no_terminate_cnt}\tremove rate : {mean_remove_rate}")

            output_dict[method]["test"] = mean_remove_rate

    with open(f"{info.out_dir}_eraser_is_attention_interpretable.json", 'w') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    # pickle.dump(remove_rate_df, open(
    #     info.faithfulness_eraser_path.split(".")[0]+".pickle", 'wb'))


def main():
    args = parse_args()
    info = pickle.load(open(args.train_info_path, "rb"))
    # モデルの都合上挿入
    info.cms_suffix = "correct_miss_separate"
    # L.set_logger(out_dir=info.out_dir, debug=args.debug)
    np.random.seed(info.seed)
    logger.info(args)
    logger.info(f"use {device}")

    # explanationデータをロード
    test_explanation_df = get_explanation_pickle_data(info, "test")

    # 評価データのサイズを指定
    eval_size = min(args.evaluate_size, len(test_explanation_df))

    # 訓練データと予測データのjsonファイルを取得
    _, test_gold_justification_list, _, test_pred_score_list, _ = get_sas_list(
        info, "test", size=eval_size) if not args.ERASER else get_eraser_list(info, 'test', size=eval_size)

    # maskの種類を選択
    if info.BERT:
        mask_id = info.vocab["[MASK]"]
    else:
        if args.mask_type == "zero":
            mask_id = info.vocab["<zero>"]
        elif args.mask_type == "rand":
            mask_id = info.vocab["<rand>"]
        else:
            raise ValueError

    # modelのロード
    # model = load_model(info) if not args.ERASER else load_model_eraser(info)
    model = load_model_eraser(info)
    model = model.to(device)
    model.print_model_info()
    logger.info(f"Load pretrained model from {info.model_path}")

    # データのロード
    (_), (_), (test_x, test_y, test_mask,
               test_attention) = load_data(info, eval_size)

    # 結果作成
    # if args.correct_miss_separate:
    #     make_faithfulness_eraser_file_correct_miss_separate(
    #         info,  test_pred_score_list, test_explanation_df, test_x, test_y, test_mask, test_attention, model)
    make_faithfulness_eraser_file(
        info, test_pred_score_list, test_explanation_df, test_x, test_y, test_mask, test_attention, model, mask_id, flip_mode=args.flip_mode)


if __name__ == '__main__':
    main()
