'''
説明
'''
import argparse
from logzero import logger
from os import path
from typing import DefaultDict, List
from glob import glob
import os
import json
from collections import defaultdict
# import sys
# sys.path.append("..")
# from getinfo import ranges
use_prompt_list = [
    "Y14_1-2_1_3",
    "Y14_1-2_2_4",
    "Y14_2-1_1_5",
    "Y14_2-1_2_3",
    "Y14_2-2_1_4",
    "Y14_2-2_2_3",
    "Y15_1-1_1_4",
    "Y15_1-3_1_2",
    # "Y15_1-3_2_4",
    # "Y15_1-3_2_5",
    "Y15_2-2_1_5",
    "Y15_2-2_2_4",
    "Y15_2-2_2_5",
    "Y15_1-3_1_5",
    "Y15_2-2_1_3"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, default="/home/hiro819/projects/JapaneseSAS/data/japanese_sas/data/Yozemi", help='input file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)

    zero_count_dict = defaultdict(lambda: defaultdict(int))
    all_count_dict = dict()
    for Y_dir in ["Y14", "Y15"]:
        for json_path in glob(f"{args.input}/{Y_dir}/*.json"):
            prompt = os.path.split(json_path)[-1].split(".")[0]
            if prompt in use_prompt_list:
                json_data = json.load(open(json_path))
                all_count_dict[prompt] = len(json_data)
                for jd in json_data:
                    for item in [chr(i) for i in range(ord("A"), ord("Z"))]:
                        key = f"{item}_Score"
                        if key in jd:
                            if jd[key] == 0:
                                zero_count_dict[prompt][item] += 1

    for prompt in zero_count_dict:
        for item in zero_count_dict[prompt]:
            print(prompt, item,
                  zero_count_dict[prompt][item], all_count_dict[prompt])


if __name__ == '__main__':
    main()
