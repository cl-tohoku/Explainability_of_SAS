'''
指定した問題のjudtificationの長さと文全体の長さを計算する
return : prompt_item, average, median, min, max, std,
'''
import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
import numpy as np

import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir', type=path.abspath, help='input dir path')
    parser.add_argument(
        '-prompt',  type=str, help='prompt')
    parser.add_argument(
        '-item',  type=str, help='item')
    # parser.add_argument(
    #     '-o', '--output', type=path.abspath, help='output file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)

    path = f"{args.input_dir}/{args.prompt}/{args.prompt[1:]}.json"

    json_data = json.load(open(path, 'r'))

    justifi_length = []
    answer_length = []
    for line in json_data:
        justifi_length.append(sum(map(int, line[f'C_{args.item}'].split(' '))))
        answer_length.append(len(line[f'C_{args.item}'].split(' ')))
    print(f"{args.prompt}_{args.item}\t{np.average(justifi_length)}\t{np.average(answer_length)}\t{np.average(justifi_length)/np.average(answer_length)}\t\
          {np.median(justifi_length)}\t{np.median(answer_length)}\t{np.median(justifi_length)/np.median(answer_length)}\t\
            {min(justifi_length)}\t{min(answer_length)}\t{min(justifi_length)/min(answer_length)}\t\
            {max(justifi_length)}\t{max(answer_length)}\t{max(justifi_length)/max(answer_length)}\t\
            {np.std(justifi_length)}\t{np.std(answer_length)}\t{np.std(justifi_length)/np.std(answer_length)}")


if __name__ == '__main__':
    main()
