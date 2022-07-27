'''
答案データのjson fileからエクセルデータを作成するコード
ついでにアテンションtestデータにおいて(1)justificationじゃないのにattentioin張られている (2)justificationなのにattentioin張られていない（３）justificationでattentioin張られている　をカウント
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

logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-o', '--output', type=path.abspath, help='output file path')
    parser.add_argument("--include_zero_score", "-izs", dest="include_zero_score", default=False,
                        action='store_true', help="By using this, zero score answer is taken into account in the justification identification")
    parser.add_argument("--train_info_path", "-info")
    parser.add_argument("--methods", '-m', nargs="*", type=str, default=None)
    parser.add_argument("--not_print_discrete",'-npd',action="store_true",default=False, help="don't print discrete justifiation")
    args = parser.parse_args()
    return args


def make_column_index(index: int) -> str:
    '''
    エクセルの列指定の文字列を生成
    27列目→AA列
    '''
    ret = []
    Z_index = ord('Z')
    A_index = ord('A')
    letter_num = ord('Z') - ord('A') + 1
    d = index + 1
    while 0 != d:
        d -= 1
        d, m = divmod(d, letter_num)
        ret.append(m)
    ret = ''.join(map(lambda c: chr(c+A_index), ret[::-1]))
    return ret


class Workbook():
    def __init__(self, output_file):
        self.workbook = xlsxwriter.Workbook(output_file)  # 新規xlsxファイル作成
        self.worksheet = dict()
        self.row = defaultdict(lambda: 1)
        self.color = self.workbook.add_format({'color': "blue", "bold": True})
        # self.color = []
        # for c in ['red', 'blue', 'green', 'orange', 'gray']:
        #     self.color.append(self.workbook.add_format(
        #         {'color': c, "bold": True}))

    def create_work_sheet(self, sheet_name, column_name_list, column_width_list):
        self.worksheet[sheet_name] = self.workbook.add_worksheet(sheet_name)
        for i, column_width in enumerate(column_width_list):
            column = chr(i+ord('A'))
            self.worksheet[sheet_name].set_column(
                f"{column}:{column}", column_width)
        self.write_row_data_to_xlsx(sheet_name, column_name_list)

    def create_answer_data_for_gold(self, answer, justification_cue, threshold):
        # assert len(answer) == len(
        #     justification_cue), f"{len(answer)}, {len(justification_cue)}\n{answer}\n{justification_cue}"
        binary_justification_cue = to_descrete_by_threshold(
            justification_cue, threshold)
        ret = []
        # max_justi = max(justification_cue)
        # assert len(answer) == len(
        #     justification_cue), f"{len(answer)},{ len(justification_cue) }"
        # print(f"{len(answer)},{ len(justification_cue) }")
        for i, token in enumerate(answer):
            if binary_justification_cue[i]:
                # if 0 < justification_cue[i] and max_justi - justification_cue[i] < threshold:
                ret.append(self.color)
            ret.append(token)
        ret.append(self.workbook.add_format({"bold": False}))
        return ret

    def create_answer_data_for_pred(self, answer, attention):
        ret = []
        max_att = max(attention)
        for att, token in zip(attention, answer):
            # 絶対色表記
            # att_color = f'{255-min(255,int(att*255)*5):0>2X}'
            # 相対色表記
            att_color = f'{255 - int((att/max_att)*255):0>2X}'
            att_color = f"#{att_color}AA{att_color}"
            font_size = (att/max_att) * 15
            ret.append(self.workbook.add_format(
                {'color': att_color, "bold": True, "font_size": font_size}))
            ret.append(token)
        return ret

    def write_row_data_to_xlsx(self, sheet_name, row_data):
        for i in range(len(row_data)):
            column = make_column_index(i)
            try:
                self.worksheet[sheet_name].write(
                    f"{column}{self.row[sheet_name]}", *row_data[i])
            except:
                self.worksheet[sheet_name].write_rich_string(
                    f"{column}{self.row[sheet_name]}", *row_data[i])
        self.row[sheet_name] += 1

    def close(self):
        self.workbook.close()


def create_discrete_justification(justification, threshold):
    '''
        連続値であるjustificationに対してthresholdで離散化する
    '''
    max_justi = max(justification)
    ret = [1 if 0 < justi and max_justi - justi <
           threshold else 0 for justi in justification]
    return ret


def print_discrete_justification_data(sentence_list, justification_list, workbook,  sheet_name: str, id: str, pred_score: int, gold_score: int, threshold=0.5):
    # assert len(sentence_list) == len(
    #     justification_list), f"{len(sentence_list)}, {len(justification_list)}"
    if len(sentence_list) != len(justification_list):
        logger.warning(
            f"length miss match,\tsentence_list:{len(sentence_list) }\tjustification_list:{len( justification_list)}")
        return
    answer = workbook.create_answer_data_for_gold(
        sentence_list, justification_list, threshold)
    row_data = [[id], answer, [pred_score], [gold_score]]
    workbook.write_row_data_to_xlsx(sheet_name, row_data)
    return


def print_continuous_justification_data(sentence_list, justification_list, workbook, sheet_name, id, pred_score, gold_score):
    # assert len(sentence_list) == len(
    #     justification_list), f"{len(sentence_list)}, {len(justification_list)}"
    if len(sentence_list) != len(justification_list):
        logger.warning(
            f"length miss match,\tsentence_list:{len(sentence_list) }\tjustification_list:{len( justification_list)}")
        return
    soft_answer = workbook.create_answer_data_for_pred(
        sentence_list, justification_list)
    row_data = [[id], soft_answer,
                [pred_score], [gold_score]]
    workbook.write_row_data_to_xlsx(sheet_name, row_data)
    return


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

    # エクセルファイル作成
    workbook = Workbook(output_file=info.explanation_xlsx_path)

    column_name_list = [["id"], ["答案"], ["pred"], ["gold"]]
    modes = ["train", 'dev', 'test']

    attention_json_file = info.justification_identification_with_zero_path if args.include_zero_score else info.justification_identification_wo_zero_path

    if not args.not_print_discrete:
        with open(attention_json_file, 'r') as f:
            justi_result_data = json.load(f)

    column_width_list = [10 for _ in range(len(column_name_list))]
    column_width_list[1] = 150  # 2番目は答案が入るので

    # sheetを作成
    for mode in modes:
        workbook.create_work_sheet(
            mode, column_name_list, column_width_list)

    # 答案，gold attentino, pred attention , hard attentinoをまとめてfor文回す

    # for train_gold_data, dev_gold_data, test_gold_data, dev_pred_data, test_pred_data in zip_longest(load_json_file(args.train_gold_json_file), load_json_file(args.dev_gold_json_file), load_json_file(args.test_gold_json_file), load_json_file(args.dev_pred_json_file), load_json_file(args.test_pred_json_file)):
    # for gold_json_file, pred_json_file, pickle_file, sheet_name in [[args.train_gold_json_file, None, None, "train"], [args.dev_gold_json_file,  args.dev_pred_json_file, args.dev_pred_pickle_file, 'dev'], [args.test_gold_json_file, args.test_pred_json_file, args.test_pred_pickle_file, 'test']]:
    for mode in ["train", 'dev', 'test']:
        for idx, (sentence_list, gold_attention_list, gold_score, pred_score, id) in enumerate(get_sas_list_for_xlsx(info, mode)):
            # goldデータを出力
            print_discrete_justification_data(
                sentence_list, gold_attention_list, workbook, mode, f"{id}-0-g", pred_score, gold_score)

            # if pred_data:
            # print(len(sentence_list))
            if mode != 'train':
                for method, maps in get_iteration_explanation_pickle_data(info, mode, idx):
                    if args.methods is None or method in args.methods:
                        # print(method, len(maps))
                        if not args.not_print_discrete:
                            print_discrete_justification_data(sentence_list, maps, workbook, mode,
                                                            f"{id}-{method}-d", pred_score, gold_score, justi_result_data[method]["test"]["threshold"])
                        print_continuous_justification_data(
                            sentence_list, maps, workbook, mode, f"{id}-{method}-c", pred_score, gold_score)

                # 空白を出力
                row_data = [[f"{id}-emp"],
                            [''], [pred_score], [gold_score]]
                workbook.write_row_data_to_xlsx(mode, row_data)

    workbook.close()


if __name__ == '__main__':
    main()
