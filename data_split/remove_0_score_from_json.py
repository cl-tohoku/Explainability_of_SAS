'''
説明
'''
import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
import json
import random
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input json file path')
    parser.add_argument("--item", type=str, help="採点項目")
    parser.add_argument(
        '-o', '--output_dir', type=path.abspath, help='output dir path')
    parser.add_argument("--zero_cnt", type=int, help="ファイルに含める0点の答案の数")
    parser.add_argument("--seed", default=0)
#    parser.add_argument('--log',
#                        default="log.txt", help='Path to log file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)
    prompt = os.path.split(args.input)[-1].split('.')[0]
    output_path = f"{args.output_dir}/{prompt}_{args.item}.json"
    assert not os.path.exists(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(args.seed)

    json_data = json.load(open(args.input))

    zero_score_list = []
    not_zero_score_list = []

    for data in json_data:
        score = data[f"{args.item}_Score"]
        if score == 0:
            zero_score_list.append(data)
        else:
            not_zero_score_list.append(data)

    random.shuffle(zero_score_list)
    zero_score_list = zero_score_list[:args.zero_cnt]

    output_list = zero_score_list + not_zero_score_list
    random.shuffle(output_list)

    json.dump(output_list, open(output_path, 'w'),
              ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
