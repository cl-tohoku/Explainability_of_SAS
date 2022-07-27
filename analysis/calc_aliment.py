'''
人間間のアノテーションのアライメントを計算するコード
input_file : 比較したい2つのファイル
output : 結果を出力するファイル

'''
import sys  # noqa
import os  # noqa
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # noqa

from sas.quadratic_weighted_kappa import quadratic_weighted_kappa
import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List, Dict, Any
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import  getinfo

def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ig', '--input_gold', type=path.abspath, help='input gold file path')
    parser.add_argument(
        '-ir', '--input_re', type=path.abspath, help='input reannotation file path')
    # parser.add_argument(
    #     '-o', '--output', type=path.abspath, help='output file dir')
    args = parser.parse_args()
    return args

def get_min_max(prompt, item):
    return getinfo.ranges[item][prompt]


def get_items(js: Dict[str, str], pref="_Score", remove_list=["EOS_Score", "Miss_Score"]) -> List[str]:
    ret = []
    for k in js.keys():
        if k.endswith(pref) and k not in remove_list:
            ret.append(k.replace(pref, ''))
    return ret


def get_score_list(json_list: List[Dict], item: str, pref="_Score") -> List[int]:
    ret = []
    for j in json_list:
        ret.append(j[item+pref])
    return ret


def get_justification_list(json_list: List[Dict], item: str) -> List[List[bool]]:
    ret = []
    for j in json_list:
        justification = list(map(lambda x: x == '1', j[item].split(' ')))
        if sum(justification) == 0:
            justification.append(True)
        else:
            justification.append(False)
        ret.append(justification)
    return ret


def test_get_justification_list():
    json_list = [{'A': "0 0 0 1 1 1"}, {'A': "1 1 1 0 0 0"}]
    target = [[False, False, False, True, True, True, False],
              [True, True, True, False, False, False, False]]
    for t, a in zip(target, get_justification_list(json_list, "A")):
        assert t == a


def preprocessed_json_data(gold_json_data: List[Dict], re_json_data: List[Dict]) -> (List[Dict], List[Dict]):
    precessed_gold_json_data = []
    precessed_re_json_data = []
    for rjd in re_json_data:
        isAppend = False
        for gjd in gold_json_data:
            if rjd["mecab"] == gjd["mecab"]:
                precessed_gold_json_data.append(gjd)
                precessed_re_json_data.append(rjd)
                isAppend = True
                break
        # if isAppend is False:
        #     logger.error(str(rjd))
        #     raise ValueError
    # assert len(re_json_data) == len(
    #     precessed_gold_json_data), f"{len(re_json_data)}, {len(precessed_gold_json_data)}"
    if len(re_json_data) != len(precessed_gold_json_data):
        logger.warning(
            f"unknown data exist. reannotation data:{len(re_json_data)}, annotation data:{len(precessed_gold_json_data)}")
    return precessed_gold_json_data, precessed_re_json_data


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)
    gold_json_data = json.load(open(args.input_gold, 'r'))
    re_json_data = json.load(open(args.input_re, 'r'))

    precessed_gold_json_data, precessed_re_json_data = preprocessed_json_data(
        gold_json_data, re_json_data)

    prompt = args.input_gold.split("/")[-1].split(".")[0]
    items = get_items(re_json_data[0])

    # QWK計算
    for item in items:
        min_rating, max_rating = get_min_max(prompt, item)
        print(min_rating, max_rating)
        re_score_list = get_score_list(precessed_re_json_data, item)
        gold_score_list = get_score_list(precessed_gold_json_data, item)
        qwk = quadratic_weighted_kappa(re_score_list, gold_score_list,min_rating=min_rating, max_rating=max_rating)
        logger.info(f"{item}(qwk) :\t{qwk}")

    # justification計算
    test_get_justification_list()
    for item in items:
        re_justification_list = get_justification_list(
            precessed_re_json_data, item)
        gold_justification_list = get_justification_list(
            precessed_gold_json_data, item)
        f1 = np.mean([precision_recall_fscore_support(g, r, average="binary")[2]
                      for g, r in zip(gold_justification_list, re_justification_list)])
        accu = np.mean([accuracy_score(g, r) for g, r in zip(
            gold_justification_list, re_justification_list)])

        logger.info(f"{item}(accu) :\t{accu}")
        logger.info(f"{item}(f1) :\t{f1}")

        # results = {"qwk": qwk, "f1": f1, "accu": accu}
        # json.dump(results, open(f"{args.output}_{item}.json", 'w'))


if __name__ == '__main__':
    main()
