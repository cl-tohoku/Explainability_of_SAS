'''
実験結果をエクセル形式で出力
設問を指定→各設問ごとの性能をシードごとに出力
指定なし　→　シードごとの性能を平均した結果を出力
'''
import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List, Dict
from glob import glob
from collections import defaultdict
import json
import numpy as np
from domain.prompt import prompts

from print_result_func import *

logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dir', type=path.abspath, help='input file dir')
    parser.add_argument("--include_zero_score", "-izs", dest="include_zero_score", default=False, action='store_true',
                        help="By using this, zero score answer is taken into account in the justification identification")
    parser.add_argument("-cms", type=str, default=None, help='正誤を分けて出力する')
    parser.add_argument("--convert_name", action="store_true", default=False)
    args = parser.parse_args()
    return args


def print_justification_identification_result_cms(dir_path: str, include_zero_score: bool, mode='correct'):
    print("prompt\ttrain size\tattn size\tseed\tjusti\tf1\trecall\tprecision")
    for prompt in get_prompt_list(dir_path):
        for item in get_item_list(dir_path, prompt):
            for train_size in get_train_size_list(dir_path, prompt, item):
                for attention_size in get_attention_size_list(dir_path, prompt, item, train_size):
                    for seed in get_seed_list(dir_path, prompt, item, train_size, attention_size):
                        prefix_file_path = f"{dir_path}/{prompt}/{item}/{train_size}/{attention_size}/{seed}/{prompt}_{item}_{train_size}_{attention_size}_{seed}"

                        f_p_file_path = f"{prefix_file_path}_justification_identification_with_zero_score_correct_miss_separate.json" if include_zero_score else f"{prefix_file_path}_justification_identification_wo_zero_score_correct_miss_separate.json"
                        if path.isfile(f_p_file_path):
                            f_p_data = json.load(
                                open(f_p_file_path, 'r'))[mode]
                            for justification_method in f_p_data:
                                if convert_name:
                                    print(
                                        f"{prompts[prompt].type}_{item}", end='\t')
                                else:
                                    print(f"{prompt}_{item}", end='\t')
                                print(train_size, end='\t')
                                print(attention_size, end='\t')
                                print(seed, end='\t')
                                print(justification_method, end='\t')

                                print(f_p_data[justification_method]
                                      ["test"]["f1"], end='\t')
                                print(f_p_data[justification_method]
                                      ["test"]["recall"], end='\t')
                                print(f_p_data[justification_method]
                                      ["test"]["precision"], end='\t')
                                print()


def print_justification_identification_result(dir_path: str, include_zero_score: bool, convert_name: bool):
    print("prompt\ttrain size\tattn size\tseed\tjusti\tf1\trecall\tprecision")
    for prompt in get_prompt_list(dir_path):
        for item in get_item_list(dir_path, prompt):
            for train_size in get_train_size_list(dir_path, prompt, item):
                for attention_size in get_attention_size_list(dir_path, prompt, item, train_size):
                    for seed in get_seed_list(dir_path, prompt, item, train_size, attention_size):
                        prefix_file_path = f"{dir_path}/{prompt}/{item}/{train_size}/{attention_size}/{seed}/{prompt}_{item}_{train_size}_{attention_size}_{seed}"

                        f_p_file_path = f"{prefix_file_path}_justification_identification_with_zero_score.json" if include_zero_score else f"{prefix_file_path}_justification_identification_wo_zero_score.json"
                        if path.isfile(f_p_file_path):
                            f_p_data = json.load(open(f_p_file_path, 'r'))
                            for justification_method in f_p_data:
                                if convert_name:
                                    print(
                                        f"{prompts[prompt].type}_{item}", end='\t')
                                else:
                                    print(f"{prompt}_{item}", end='\t')
                                print(train_size, end='\t')
                                print(attention_size, end='\t')
                                print(seed, end='\t')
                                print(justification_method, end='\t')

                                print(f_p_data[justification_method]
                                      ["test"]["f1"], end='\t')
                                print(f_p_data[justification_method]
                                      ["test"]["recall"], end='\t')
                                print(f_p_data[justification_method]
                                      ["test"]["precision"], end='\t')
                                print()


def main():
    args = parse_args()
    logger.info(args)
    if args.cms == None:
        print_justification_identification_result(
            args.dir, args.include_zero_score, args.convert_name)
    else:
        print_justification_identification_result_cms(
            args.dir, args.include_zero_score, args.cms, args.convert_name)


if __name__ == '__main__':
    main()
