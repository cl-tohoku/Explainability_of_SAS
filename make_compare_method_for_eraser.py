'''
attentionとある手法での削除率を比較してexcelファイルにするコード
'''

import argparse
import logzero
from logzero import logger
import logging
from os import path
from collections import defaultdict
from itertools import zip_longest
from typing import List
import json
import os
import xlsxwriter
import pickle

import numpy as np
from getinfo import TrainInfo
from analysis.util import *
from make_explanation_xlsx import *

logger.setLevel(logging.DEBUG)


def print_continuous_justification_data_for_compare(sentence_list, justification_list, workbook, sheet_name, id, pred_score, gold_score, remove_ratio, compare_remove_ratio):
    assert len(sentence_list) == len(justification_list)
    soft_answer = workbook.create_answer_data_for_pred(
        sentence_list, justification_list)
    row_data = [[id], soft_answer,
                [pred_score], [gold_score], [remove_ratio], [compare_remove_ratio]]
    workbook.write_row_data_to_xlsx(sheet_name, row_data)
    return


def print_discrete_justification_data_for_compare(sentence_list, justification_list, workbook,  sheet_name: str, id: str, pred_score: int, gold_score: int, remove_rate: float, threshold=0.5, ):
    assert len(sentence_list) == len(justification_list)
    answer = workbook.create_answer_data_for_gold(
        sentence_list, justification_list, threshold)
    row_data = [[id], answer, [pred_score],
                [gold_score], [None], [remove_rate]]
    workbook.write_row_data_to_xlsx(sheet_name, row_data)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_info_path", "-info")
    parser.add_argument(
        "--method", '-m', default='Integrated_Gradients', help='attentionと比較したい手法')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)
    info = pickle.load(open(args.train_info_path, "rb"))

    if info.BERT:
        attention_key = f"C_{info.item}"
        answer_key = "Char"
    else:
        attention_key = f"{info.item}"
        answer_key = "mecab"

    attention_key = "Attention_Weights"

    # エクセルファイル作成
    workbook = Workbook(output_file=info.explanation_xlsx_path.split(
        ".")[0] + f"_compare_attention_and_{args.method}.xlsx")

    column_name_list = [["id"], ["答案"], [
        "pred"], ["gold"], ['削除率'], ["削除率の比較値"]]
    modes = ['test']

    column_width_list = [10 for _ in range(len(column_name_list))]
    column_width_list[1] = 150  # 2番目は答案が入るので

    # sheetを作成
    for mode in modes:
        workbook.create_work_sheet(
            mode, column_name_list, column_width_list)

    # remove_ratio_dfをロード
    remove_rate_df = pickle.load(
        open(info.faithfulness_eraser_path.split(".")[0]+".pickle", 'rb'))

    # for train_gold_data, dev_gold_data, test_gold_data, dev_pred_data, test_pred_data in zip_longest(load_json_file(args.train_gold_json_file), load_json_file(args.dev_gold_json_file), load_json_file(args.test_gold_json_file), load_json_file(args.dev_pred_json_file), load_json_file(args.test_pred_json_file)):
    # for gold_json_file, pred_json_file, pickle_file, sheet_name in [[args.train_gold_json_file, None, None, "train"], [args.dev_gold_json_file,  args.dev_pred_json_file, args.dev_pred_pickle_file, 'dev'], [args.test_gold_json_file, args.test_pred_json_file, args.test_pred_pickle_file, 'test']]:
    for mode in modes:
        for idx, (sentence_list, gold_attention_list, gold_score, pred_score, id) in enumerate(get_sas_list_for_xlsx(info, mode)):
            compare_remove_ratio = remove_rate_df[attention_key][idx] - \
                remove_rate_df[args.method][idx]
            # goldデータを出力
            print_discrete_justification_data_for_compare(
                sentence_list, gold_attention_list, workbook, mode, f"{id}-0-g", pred_score, gold_score, compare_remove_ratio)

            # if pred_data:
            # print(len(sentence_list))
            if mode != 'train':
                for method, maps in get_iteration_explanation_pickle_data(info, mode, idx):
                    # print(method, len(maps))
                    if method in [attention_key, args.method]:
                        print_continuous_justification_data_for_compare(
                            sentence_list, maps, workbook, mode, f"{id}-{method}-c", pred_score, gold_score, remove_rate_df[method][idx], compare_remove_ratio)

                # # 空白を出力
                # row_data = [[f"{id}-emp"],
                #             [''], [pred_score], [gold_score]]
                # workbook.write_row_data_to_xlsx(mode, row_data)

    workbook.close()


if __name__ == '__main__':
    main()
