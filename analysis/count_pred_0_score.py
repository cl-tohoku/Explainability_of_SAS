'''
予測結果で0点になってる数をカウントする
'''
import sys
sys.path.append("..")

import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
import pickle
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_info_path", "-info")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    # logger.info(args)
    info = pickle.load(open(args.train_info_path, 'rb'))
    pred_json_path = f"{info.out_dir}_test.json"
    json_data = json.load(open(pred_json_path))
    item_key = f"{info.item}_score"

    zero_count = sum([jd[item_key] == 0 for jd in json_data])
    print(f"{info.prompt_name}_{info.item}", zero_count, sep='\t')


if __name__ == '__main__':
    main()
