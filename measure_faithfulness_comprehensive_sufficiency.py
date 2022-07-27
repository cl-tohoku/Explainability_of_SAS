
'''
A Benchmark to Evaluate Rationalized NLP Models で提唱されているfaithfulnessを計算する手法
'''
import pandas as pd
import numpy as np
import argparse

from torch.random import initial_seed
from util import logger as L
from sas import util
from typing import List
import pickle
import json
import torch
import torch.nn.functional as F
from logging import getLogger
from interpretability.explanation import Explanation
from analysis.util import *
from tqdm import tqdm
import time
from collections import defaultdict
logger = getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_zero_score", "-izs", dest="include_zero_score", default=False,
                        action='store_true', help="By using this, zero score answer is taken into account in the justification identification")
    parser.add_argument("--train_info_path", "-info")
    parser.add_argument("--print-debug-info", "-debug",
                        dest="debug", default=False, action='store_true', help="")
    parser.add_argument("-ERASER", action='store_true',
                        default=False, help='use eraser dataset')
    args = parser.parse_args()
    return args


def create_comprehensiveness_masked_ids(x, justifications, k, mask_id, bert=False):
    masked_x = torch.clone(x).detach()
    OFF_SET = 1 if bert else 0
    for i in range(x.size(0)):
        k_tokens = int(len(justifications[i]) * k)
        top_k_tokens = torch.topk(torch.tensor(justifications[i]), k_tokens)
        masked_x[i][top_k_tokens.indices + OFF_SET] = mask_id
    return masked_x


def create_sufficiency_masked_ids(x, justifications, k, mask_id, pad_id, cls_id=None, sep_id=None, bert=False):
    masked_x = torch.clone(x).detach()
    comprehensiveness_masked_x = create_comprehensiveness_masked_ids(
        x, justifications, 1 - k, mask_id, bert)
    masked_x[(comprehensiveness_masked_x != mask_id) & (masked_x != pad_id)
             & (masked_x != sep_id) & (masked_x != cls_id)] = mask_id
    return masked_x


def predict(info, knn_model, output, index=None):
    if info.metric.endswith("Triplet Loss"):
        output = output.to("cpu").numpy()
        _, probs = knn_model.predict(output, is_trust_score=True)
        # probs = []
        # for i in range(len(output)):
        # pos_dis, neg_dist = knn_model.get_pos_neg_dist(output[i])
        # probs.append(knn_model.initial_dist - (pos_dis - neg_dist))

        # probs.append(prob)
        # probs = knn_model.get_probs(output)
        probs = torch.tensor(probs)
        return probs, None
    elif info.metric == "crossentropy":
        probs = F.softmax(output, dim=-1)
        _, probs_index = torch.max(probs, 1)
        probs_index = probs_index.reshape(-1, 1)
        if index is None:
            probs = torch.gather(probs, 1, probs_index)
        else:
            probs = torch.gather(probs, 1, index)
        return probs, probs_index
    else:
        raise ValueError(
            "we haven't inplement other metrics yet")


def calc_sufficiency(info, x, x_mask, justifications, model, special_token_id, k_list=[0.01, 0.05, 0.1, 0.2, 0.5], batch_size=64, bert=False, knn_model=None):
    sufficiency_list = []
    model.eval()
    with torch.no_grad():
        for k in k_list:
            _sufficiency_list = []
            for batch_index in range(0, len(x), batch_size):
                batch_x = x[batch_index:batch_index + batch_size]
                batch_j = justifications[batch_index:batch_index + batch_size]
                batch_sufficiency_masked_x = create_sufficiency_masked_ids(
                    batch_x, batch_j, k, mask_id=special_token_id["mask"], pad_id=special_token_id["pad"], cls_id=special_token_id["cls"], sep_id=special_token_id["sep"], bert=bert)

                batch_x = batch_x.to(device)
                batch_m = x_mask[batch_index:batch_index +
                                 batch_size].to(device)
                batch_sufficiency_masked_x = batch_sufficiency_masked_x.to(
                    device)

                # _, probs = knn_model.predict(model.predict(
                #         ids=batch_x, attention_mask=batch_m), is_trust_score=True)
                # elif info.metric == "crossentropy":
                #     probs = F.softmax(model.predict(
                #         ids=batch_x, attention_mask=batch_m), dim=-1)
                # else:
                #     raise ValueError(
                #         "we haven't inplement other metrics yet")
                output = model.predict(
                    ids=batch_x, attention_mask=batch_m)

                if info.metric.endswith("Triplet Loss"):
                    # マスクをしない段階での答案ベクトルを取得しsetする
                    knn_model.set_pos_neg_instance(output.to("cpu"))

                probs, probs_index = predict(info, knn_model, output)
                # print("before", probs)

                # マスクした入力を取得
                sufficiency_output = model.predict(
                    ids=batch_sufficiency_masked_x, attention_mask=batch_m)
                sufficiency_probs, _ = predict(
                    info, knn_model, sufficiency_output, probs_index)

            # print("after", sufficiency_probs)

                sufficiency = (
                    probs - sufficiency_probs).mean().detach().item()
                _sufficiency_list.append(sufficiency)

                if info.metric.endswith("Triplet Loss"):
                    knn_model.reset_pos_neg_instance()

            sufficiency_list.append(np.mean(_sufficiency_list))

    sufficiency = np.mean(sufficiency_list)
    return sufficiency


def calc_comprehensiveness(info, x, x_mask, justifications, model, special_token_id, k_list=[0.01, 0.05, 0.1, 0.2, 0.5], batch_size=64, bert=False, knn_model=None):
    comprehensiveness_list = []
    model.eval()
    with torch.no_grad():
        for k in k_list:
            _comprehensiveness_list = []
            for batch_index in range(0, len(x), batch_size):
                batch_x = x[batch_index:batch_index + batch_size]
                batch_j = justifications[batch_index:batch_index + batch_size]
                batch_comprehensiveness_masked_x = create_comprehensiveness_masked_ids(
                    batch_x, batch_j, k, mask_id=special_token_id["mask"], bert=bert)

                batch_x = batch_x.to(device)
                batch_m = x_mask[batch_index:batch_index +
                                 batch_size].to(device)
                batch_comprehensiveness_masked_x = batch_comprehensiveness_masked_x.to(
                    device)

                # probs = F.softmax(model.predict(
                #     ids=batch_x, attention_mask=batch_m), dim=-1)
                # _, probs_index = torch.max(probs, 1)
                output = model.predict(
                    ids=batch_x, attention_mask=batch_m)

                if info.metric.endswith("Triplet Loss"):
                    # マスクをしない段階での答案ベクトルを取得しsetする
                    knn_model.set_pos_neg_instance(output.to("cpu"))

                probs, probs_index = predict(info, knn_model, output)

                # マスクした入力を取得
                # comprehensiveness_probs = F.softmax(model.predict(
                #     ids=batch_comprehensiveness_masked_x, attention_mask=batch_m), dim=-1)
                comprehensiveness_output = model.predict(
                    ids=batch_comprehensiveness_masked_x, attention_mask=batch_m)
                comprehensiveness_probs, _ = predict(
                    info, knn_model, comprehensiveness_output, probs_index)

                comprehensiveness = (
                    probs - comprehensiveness_probs).mean().detach().item()
                _comprehensiveness_list.append(comprehensiveness)
            comprehensiveness_list.append(np.mean(_comprehensiveness_list))

    comprehensiveness = np.mean(comprehensiveness_list)
    return comprehensiveness


def test_create_comprehensiveness_masked_ids():
    mask_id = 101
    pad_id = 999
    sep_id = 1
    cls_id = 0
    x = torch.LongTensor([[cls_id, 2, 2, 3, 4, 5, sep_id, pad_id, pad_id, pad_id], [
                         cls_id, 2, 4, 5, 7, 3, 5, 3, sep_id, pad_id]])
    answers = torch.LongTensor([[cls_id, mask_id, mask_id, 3, 4, 5, sep_id, pad_id, pad_id, pad_id],
                                [cls_id, 2, 4, 5, mask_id, 3, mask_id, mask_id, sep_id, 999]])
    justifications = [[5, 4, 3, 2, 1], [cls_id, 2, 3, 6, 1, 5, 4]]
    k = 0.5
    results = create_comprehensiveness_masked_ids(
        x, justifications, k, mask_id=mask_id, bert=True)
    assert torch.all(answers == results), f"{answers}\n{results}"


def test_create_sufficiency_masked_ids():
    mask_id = 101
    pad_id = 999
    sep_id = 1
    cls_id = 0
    x = torch.LongTensor([[cls_id, 2, 2, 3, 4, 5, sep_id, pad_id, pad_id, pad_id], [
                         cls_id, 2, 4, 5, 7, 3, 5, 3, sep_id, pad_id]])
    justifications = [[5, 4, 3, 2, 1], [0, 2, 3, 6, 1, 5, 4]]
    k = 0.5
    answers = torch.LongTensor([[cls_id, 2, 2, mask_id, mask_id, mask_id, sep_id, pad_id, pad_id, pad_id], [
                               cls_id, mask_id, mask_id, mask_id, 7, mask_id, 5, 3, sep_id, pad_id]])
    results = create_sufficiency_masked_ids(
        x, justifications, k, mask_id=mask_id, cls_id=cls_id, pad_id=pad_id, sep_id=sep_id, bert=True)
    assert torch.all(answers == results), f"{answers}\n{results}"


def main():
    args = parse_args()
    info = pickle.load(open(args.train_info_path, "rb"))
    # モデルの都合上挿入
    info.faithful_comprehensiveness_sufficiency = "comprehensiveness_sufficiency"
    # L.set_logger(out_dir=info.out_dir, debug=args.debug)
    np.random.seed(info.seed)
    logger.info(args)
    logger.info(f"use {device}")

    # 訓練データと予測データのjsonファイルを取得
    _, test_gold_justification_list, _, test_pred_score_list, _ = get_sas_list(
        info, "test") if not args.ERASER else get_eraser_list(info, 'test')

    # explanationデータをロード
    test_explanation_df = get_explanation_pickle_data(info, "test")

    # modelのロード
    # model = load_model(info) if not args.ERASER else load_model_eraser(info)
    model = load_model_eraser(info)

    model = model.to(device)
    model.print_model_info()
    logger.info(f"Load pretrained model from {info.model_path}")

    # データのロード
    (_), (_), (test_x, _, test_mask, _) = load_data(info)

    special_token_id = get_special_token_id(info)

    # knn_modelを渡す
    if info.metric.endswith("Triplet Loss"):
        from interpretability.knn_model import KnnModel
        from make_knn_xlsx import load_pickle
        train_data, train_label, * \
            _ = load_pickle(f"{info.out_dir}_train_result.pickle")
    knn_model = KnnModel(train_data, train_label) if info.metric.endswith(
        "Triplet Loss") else None

    outputs = dict()
    for method in test_explanation_df.columns:
        print(method)
        test_comprehensiveness = calc_comprehensiveness(
            info, test_x, test_mask, test_explanation_df[method].tolist(), model, special_token_id, knn_model=knn_model)
        print("compreehensiveness: ", test_comprehensiveness)

        test_sufficiency = calc_sufficiency(
            info, test_x, test_mask, test_explanation_df[method].tolist(), model, special_token_id, knn_model=knn_model)
        print("sufficiency: ", test_sufficiency)
        outputs[method] = {"test": {"sufficiency": test_sufficiency,
                                    "comprehensiveness": test_comprehensiveness}}
        # save_data_to_json({"test":{"sufficiency":sufficiency, "comprehensiveness":comprehensiveness}}, file_path)

    json.dump(outputs, open(
        f"{info.out_dir}_sufficiency_comprehensive.json", 'w'), indent=2)


if __name__ == '__main__':
    main()
