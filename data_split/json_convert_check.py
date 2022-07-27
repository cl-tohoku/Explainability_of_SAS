'''
json形式に変換したデータが変なことになっていないかチェックするためのコード
CRLEAのデータはエクセル形式で得点がまとめられているので，それを用いてjsonのデータのチェックを行う
'''


import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
from collections import defaultdict
import json

log_format = '%(color)s[%(levelname)1.1s ]%(end_color)s %(message)s'
formatter = logzero.LogFormatter(fmt=log_format)
logzero.formatter(formatter)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ij', '--input_json', type=path.abspath, help='input json file path')
    parser.add_argument(
        '-it', '--input_tsx', type=path.abspath, help='input tsx file path')
    parser.add_argument(
        '-log', type=path.abspath, help='output log file path')
    parser.add_argument(
        "-need",  nargs="*", type=str, help="neccesary item point (この項目が0点だと全体点が0点になる)", default=None)
    parser.add_argument("-cos", "--check_overall_score", default=False,
                        action='store_true', help="check all over score")
#    parser.add_argument('--log',
#                        default="log.txt", help='Path to log file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logzero.logfile(args.log, mode='w')  # 追加: logfileの作成

    # 答案をキーとして得点のリストを返す辞書
    # dict([A,B,C,...,Deduction,All_Score])の順番
    score_dict = defaultdict(list)
    excel_id_dict = defaultdict(int)
    with open(args.input_tsx, 'r') as f:
        # 最初の行は説明文なので削除
        _ = f.readline()
        for line in f:
            id, text, *score_list = line.rstrip().split('\t')
            # 空文字を0点に置き換える
            score_list = [int(s) if s else 0 for s in score_list]
            score_dict[text] = score_list
            excel_id_dict[text] = int(id)

    json_data = json.load(open(args.input_json, 'r'))

    for j_d in json_data:
        text = j_d['Char'].replace(' ', '')
        # assert text in score_dict.keys(
        # ), f"[{j_d['id']}]エクセルデータに存在しません : {text}"
        if not text in score_dict.keys():
            logger.error(f"[{j_d['id']}]エクセルデータに存在しません : {text}")

        if args.check_overall_score:
            # assert j_d['score'] == int(
            #     score_dict[text][-1]), f"[{j_d['id']}] 全体点が違います．excel:{j_d['score']}, json:{score_dict[text][-1]}"
            if not j_d['score'] == int(score_dict[text][-1]):
                logger.error(
                    f"[excel id:{excel_id_dict[text]:>4} json id:{j_d['id']:>4}]\t全体点が違います\texcel:{score_dict[text][-1]:>4}, json:{j_d['score']:>4}")
        idx = 0
        for k, v in j_d.items():
            if score_dict[text][idx] == '':
                print(score_dict[text])
                raise ValueError
            if k.endswith("_Score") and k not in ["Miss_Score", "EOS_Score"]:
                # assert v == int(
                #     score_dict[text][idx]), f"[{j_d['id']}]\t{k}の得点が違います\texcel:{score_dict[text][idx]}, json:{v}"
                if not v == int(score_dict[text][idx]):
                    logger.error(
                        f"[excel id:{excel_id_dict[text]:>4} json id:{j_d['id']:>4}]\t{k}の得点が違います\texcel:{score_dict[text][idx]:>4}, json:{v:>4}")
                idx += 1
    return


if __name__ == '__main__':
    main()
