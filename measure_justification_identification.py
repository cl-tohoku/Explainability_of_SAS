'''
アテンションについてf値，precision, recall. lossを計算．
f値などはdev setによって閾値を決定し，計算．
lossは(gold - pred) ** 2　の単語平均を計算
教師なしの時と教師ありの時では0点答案で顕著な差が出るので，なくす
2021/01/21：attentionで評価するか，attention*normで評価するかを選択可能にした
python attention_evaluate.py -trgj /work01/tasuku/project/JapaneseSAS_BERT/data/japanese_sas/data_rev/data_acl-bea19/data_acl-bea19/Y14_2-2_1_4/data_size/Y14_2-2_1_4_train.200.0.json -dvgj /work01/tasuku/project/JapaneseSAS_BERT/data/japanese_sas/data_rev/data_acl-bea19/data_acl-bea19/Y14_2-2_1_4/Y14_2-2_1_4_dev.json -tegj /work01/tasuku/project/JapaneseSAS_BERT/data/japanese_sas/data_rev/data_acl-bea19/data_acl-bea19/Y14_2-2_1_4/Y14_2-2_1_4_test.json -dpj /work01/tasuku/project/JapaneseSAS_BERT/results/embedding_per_epoch_for_word2vec/Y14_2-2_1_4/200/0/A/triplet/Y14_2-2_1_4_dev.json -tepj /work01/tasuku/project/JapaneseSAS_BERT/results/embedding_per_epoch_for_word2vec/Y14_2-2_1_4/200/0/A/triplet/Y14_2-2_1_4_test.json --tokenize mecab
'''

from analysis.util import *
import numpy as np
from sklearn.metrics import precision_recall_curve
from torch.utils import data
import torch.nn.functional as F
import argparse
from util import logger as L
from sas import handling_data, util
from os import path
from typing import List
import pickle
import json
import MeCab
import torch
from getinfo import TrainInfo
import pickle
from logging import getLogger
import getinfo
from Dataset import part_scoring_set
from collections import defaultdict
import pandas as pd
from copy import deepcopy
logger = getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_zero_score", "-izs", dest="include_zero_score", default=False,
                        action='store_true', help="By using this, zero score answer is taken into account in the justification identification")
    parser.add_argument("--train_info_path", "-info")
    parser.add_argument("--print-debug-info", "-debug",
                        dest="debug", default=False, action='store_true', help="")
    parser.add_argument("--correct_miss_separate", "-cms", dest="correct_miss_separate", default=False,
                        action='store_true', help="根拠箇所推定の評価の時に得点予測を間違ったものとあってたもので分けて評価する")
    args = parser.parse_args()
    return args


def calc_pre_rec_f1(gold_attention_list: List, pred_attention_list: List, gold_score_list: List, threshold: float, include_norm_for_justification: bool, include_zero_score: bool):

    assert len(gold_attention_list) == len(
        pred_attention_list), f"{len(gold_attention_list)}, {len(pred_attention_list)}"

    # まずアテンションんお最大値を求める
    max_a_list = list(map(max, pred_attention_list))
    precision_list, recall_list, f1_list = [], [], []
    for gal, pal, gsl, max_a in zip(gold_attention_list, pred_attention_list, gold_score_list, max_a_list):
        TP, TN, FP, FN = 0, 0, 0, 0
        # goldが0点の答案は除外する
        if not include_zero_score and gsl == 0:
            continue
        # assert len(gal) == len(pal), f"{len(gal)}, {len(pal)}"
        for gold_att, pred_att in zip(gal, pal):
            if gold_att == 1:
                if 0 < pred_att and max_a - pred_att < threshold:
                    TP += 1
                else:
                    TN += 1
            else:
                if 0 < pred_att and max_a - pred_att < threshold:
                    FN += 1
                else:
                    FP += 1
        precision = TP/(TP+FN) if TP+FN != 0 else 0
        recall = TP/(TP+TN) if TP+TN != 0 else 0
        f1 = (2*precision*recall)/(precision +
                                   recall) if precision+recall != 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    return np.mean(precision_list), np.mean(recall), np.mean(f1)


def calc_pre_rec_f1_overall(gold_attention_list: List, pred_attention_list: List, threshold: float):
    # まずアテンションんお最大値を求める
    max_a_list = list(map(max, pred_attention_list))
    TP, TN, FP, FN = 0, 0, 0, 0
    for gal, pal, max_a in zip(gold_attention_list, pred_attention_list, max_a_list):
        # goldが0点の答案は除外する
        # assert len(gal) == len(pal), f"{len(gal)}, {len(pal)}"
        for i, (gold_att, pred_att) in enumerate(zip(gal, pal)):
            if gold_att == 1:
                if 0 < pred_att and max_a - pred_att < threshold:
                    TP += 1
                else:
                    TN += 1
            else:
                if 0 < pred_att and max_a - pred_att < threshold:
                    FN += 1
                else:
                    FP += 1
    precision = TP/(TP+FN) if TP+FN != 0 else 0
    recall = TP/(TP+TN) if TP+TN != 0 else 0
    f1 = (2*precision*recall)/(precision +
                               recall) if precision+recall != 0 else 0
    return precision, recall, f1


def get_best_f1(dev_gold_justification_list, test_gold_justification_list, dev_pred_justification_list, test_pred_justification_list):
    # logger.debug(dev_pred_attention_list)
    # devデータを元に閾値を決定
    dev_threshold, dev_f1, dev_precision, dev_recall = 0, 0, 0, 0
    for tmp_threshold in np.arange(0, 1, 0.001):
        # precisionを計算
        tmp_precision, tmp_recall, tmp_f1 = calc_pre_rec_f1_overall(
            dev_gold_justification_list, dev_pred_justification_list, tmp_threshold)
        if dev_f1 < tmp_f1:
            dev_threshold, dev_f1, dev_precision, dev_recall = tmp_threshold, tmp_f1, tmp_precision, tmp_recall
    # testデータでjustification identificationを計算
    test_precision, test_recall, test_f1 = calc_pre_rec_f1_overall(
        test_gold_justification_list, test_pred_justification_list, dev_threshold)
    return (dev_f1, dev_recall, dev_precision, dev_threshold), (test_f1, test_recall, test_precision, dev_threshold)
    # -------------------------------------ここからmeasure faithfulness


def get_correct_answer_index_list(gold_score_list, pred_score_list):
    assert len(gold_score_list) == len(pred_score_list)

    correct_index_list = []
    miss_index_list = []
    for i, (gs, ps) in enumerate(zip(gold_score_list, pred_score_list)):
        if gs == ps:
            correct_index_list.append(i)
        else:
            miss_index_list.append(i)

    return {"correct": set(correct_index_list), "miss": set(miss_index_list)}


def make_justification_identification_file_correct_miss_separate(info, dev_gold_justification_list, dev_gold_score_list, dev_explanation_df, test_gold_justification_list, test_gold_score_list, test_explanation_df, include_zero_score, dev_pred_score_list=None, test_pred_score_list=None):
    master_dev_gold_justification_list = deepcopy(dev_gold_justification_list)
    master_test_gold_justification_list = deepcopy(
        test_gold_justification_list)
    # 得点予測があってるものと間違ってるもので区別する
    output_dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))

    # 予測があっている答案のindexを取得
    dev_answer_idx_dict = get_correct_answer_index_list(
        dev_gold_score_list, dev_pred_score_list)
    test_answer_idx_dict = get_correct_answer_index_list(
        test_gold_score_list, test_pred_score_list)

    for method in dev_explanation_df.columns:
        for mode in dev_answer_idx_dict.keys():

            dev_pred_justification_list = dev_explanation_df[method][dev_answer_idx_dict[mode]]
            dev_gold_justification_list = [[flag for flag in dgjl] for i, dgjl in enumerate(
                master_dev_gold_justification_list) if i in dev_answer_idx_dict[mode]]
            test_pred_justification_list = test_explanation_df[method][test_answer_idx_dict[mode]]
            test_gold_justification_list = [[flag for flag in tgjl] for i, tgjl in enumerate(
                master_test_gold_justification_list) if i in test_answer_idx_dict[mode]]

            # zero scoreを削除
            if not include_zero_score:
                dev_gold_justification_list, dev_pred_justification_list = exclude_zero_score_answer(
                    dev_gold_justification_list, dev_pred_justification_list, dev_gold_score_list)
                test_gold_justification_list, test_pred_justification_list = exclude_zero_score_answer(
                    test_gold_justification_list, test_pred_justification_list, test_gold_score_list)

            if len(dev_gold_justification_list) == 0 or len(test_gold_justification_list) == 0:
                (dev_f1, dev_recall, dev_precision, dev_threshold), (test_f1, test_recall,
                                                                     test_precision, test_threshold) = (None, None, None, None), (None, None, None, None)
            else:
                (dev_f1, dev_recall, dev_precision, dev_threshold), (test_f1, test_recall, test_precision, test_threshold) = get_best_f1(
                    dev_gold_justification_list, test_gold_justification_list, dev_pred_justification_list, test_pred_justification_list)

            logger.info(
                f"{method} {mode} cnt:{len(test_gold_justification_list)} test_f1: {test_f1}")

            output_dict[mode][method]['dev'] = {"f1": dev_f1, "precision": dev_precision,   "recall": dev_recall,
                                                "threshold": dev_threshold, "cnt": len(dev_gold_justification_list)}
            output_dict[mode][method]['test'] = {"f1": test_f1, "precision": test_precision,   "recall": test_recall,
                                                 "threshold": test_threshold, "cnt": len(test_gold_justification_list)}

    output_path = info.justification_identification_with_zero_path if include_zero_score else info.justification_identification_wo_zero_path
    output_path = f"{output_path.split('.')[0]}_{info.cms_suffix}.json"
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)


def make_justification_identification_file(info, dev_gold_justification_list, dev_gold_score_list, dev_explanation_df, test_gold_justification_list, test_gold_score_list, test_explanation_df, include_zero_score):
    output_dict = defaultdict(lambda: defaultdict(float))
    master_dev_gold_justification_list = deepcopy(dev_gold_justification_list)
    master_test_gold_justification_list = deepcopy(
        test_gold_justification_list)

    for method in dev_explanation_df.columns:
        dev_pred_justification_list = dev_explanation_df[method]
        test_pred_justification_list = test_explanation_df[method]
        # zero scoreを削除
        if not include_zero_score:
            dev_gold_justification_list, dev_pred_justification_list = exclude_zero_score_answer(
                master_dev_gold_justification_list, dev_pred_justification_list, dev_gold_score_list)
            test_gold_justification_list, test_pred_justification_list = exclude_zero_score_answer(
                master_test_gold_justification_list, test_pred_justification_list, test_gold_score_list)

        (dev_f1, dev_recall, dev_precision, dev_threshold), (test_f1, test_recall, test_precision, test_threshold) = get_best_f1(
            dev_gold_justification_list, test_gold_justification_list, dev_pred_justification_list, test_pred_justification_list)

        logger.info(f"{method} test_f1: {test_f1}")

        output_dict[method]['dev'] = {"f1": dev_f1, "precision": dev_precision,   "recall": dev_recall,
                                                    "threshold": dev_threshold}
        output_dict[method]['test'] = {"f1": test_f1, "precision": test_precision,   "recall": test_recall,
                                       "threshold": test_threshold}

    output_path = info.justification_identification_with_zero_path if include_zero_score else info.justification_identification_wo_zero_path
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    logger.info(args)
    info = pickle.load(open(args.train_info_path, "rb"))
    # モデルの都合上挿入
    info.cms_suffix = "correct_miss_separate"
    # L.set_logger(out_dir=info.out_dir, debug=args.debug)
    np.random.seed(info.seed)

    # 訓練データと予測データのjsonファイルを取得
    _, dev_gold_justification_list, dev_gold_score_list, dev_pred_score_list, _ = get_sas_list(
        info, "dev")
    _, test_gold_justification_list, test_gold_score_list, test_pred_score_list, _ = get_sas_list(
        info, "test")

    # explanationデータをロード
    dev_explanation_df = get_explanation_pickle_data(info, "dev")
    test_explanation_df = get_explanation_pickle_data(info, "test")

    # 結果作成
    if args.correct_miss_separate:
        make_justification_identification_file_correct_miss_separate(
            info, dev_gold_justification_list, dev_gold_score_list, dev_explanation_df, test_gold_justification_list, test_gold_score_list, test_explanation_df, args.include_zero_score, dev_pred_score_list, test_pred_score_list)
    make_justification_identification_file(
        info, dev_gold_justification_list, dev_gold_score_list, dev_explanation_df, test_gold_justification_list, test_gold_score_list, test_explanation_df, args.include_zero_score)


if __name__ == '__main__':
    main()
