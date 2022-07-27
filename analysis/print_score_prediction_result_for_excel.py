"""
採点実験結果をエクセル形式で出力
"""
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
    parser.add_argument("--suffix", type=str, default="evaluation_result.json")
    parser.add_argument("--convert_name", action="store_true", default=False)
    args = parser.parse_args()
    return args


def print_prediction_result(dir_path: str, suffix: str, convert_name: bool):
    print("prompt\ttrain size\tattn size\tseed\tqwk\tmse")
    for prompt in get_prompt_list(dir_path):
        for item in get_item_list(dir_path, prompt):
            for train_size in get_train_size_list(dir_path, prompt, item):
                for attention_size in get_attention_size_list(dir_path, prompt, item, train_size):
                    for seed in get_seed_list(dir_path, prompt, item, train_size, attention_size):
                        prefix_file_path = f"{dir_path}/{prompt}/{item}/{train_size}/{attention_size}/{seed}/{prompt}_{item}_{train_size}_{attention_size}_{seed}"

                        evaluation_result_file_path = f"{prefix_file_path}_{suffix}"

                        if path.isfile(evaluation_result_file_path):
                            try:
                                with open(evaluation_result_file_path, 'r') as f:
                                    json_data = json.load(f)
                                if convert_name:
                                    print(
                                        f"{prompts[prompt].type}_{item}", end='\t')
                                else:
                                    print(f"{prompt}_{item}", end='\t')

                                print(train_size, end='\t')
                                print(attention_size, end='\t')
                                print(seed, end='\t')
                                print(json_data["test_qwk"], end='\t')
                                print(json_data["test_mse"][0], end='\t')
                            except:
                                raise ValueError(
                                    f"file open error {evaluation_result_file_path}")
                        else:
                            continue

                        print()


def main():
    args = parse_args()
    logger.info(args)

    print_prediction_result(args.dir, args.suffix, args.convert_name)


if __name__ == '__main__':
    main()
