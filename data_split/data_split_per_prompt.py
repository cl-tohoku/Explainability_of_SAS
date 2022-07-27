'''
jsonデータを訓練データ，開発データ，テストデータに分割します．
'''
import argparse
import logzero
from logzero import logger
import logging
from os import path
import os
from typing import List
import json
from glob import glob
import random

logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input json file')
    parser.add_argument(
        '-o', '--output', type=path.abspath, help='output dir path')
    parser.add_argument(
        '-trs', '--train_size', type=int, help='train data size')
    parser.add_argument(
        '-ds', '--dev_size', type=int, help='dev data size')
    parser.add_argument(
        '-tes', '--test_size', type=int, help='train data size')
    parser.add_argument("--seed", type=int, help="seed")
    # parser.add_argument("-split", nargs="*", type=int,
    #                     help="split size", default=[50, 100, 200, 400, 800, 1600])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(args)

    # jsonデータを取得
    prompt = os.path.split(args.input)[-1].replace('.json', "")
    # train, dev, test に分割
    with open(args.input, 'r') as fin:
        json_data = json.load(fin)
    # shuffle
    random.shuffle(json_data)

    os.makedirs(f"{args.output}/{prompt}", exist_ok=True)
    train_output_path = f"{args.output}/{prompt}/{prompt}_train.{args.train_size}.{args.seed}.json"
    dev_output_path = f"{args.output}/{prompt}/{prompt}_dev.{args.seed}.json"
    test_output_path = f"{args.output}/{prompt}/{prompt}_test.{args.seed}.json"

    with open(train_output_path, 'w') as fo_train, open(dev_output_path, 'w') as fo_dev, open(test_output_path, 'w') as fo_test:
        print(json.dumps(json_data[:args.train_size],
                         indent=4, ensure_ascii=False), file=fo_train)
        print(json.dumps(json_data[args.train_size:args.train_size + args.test_size],
                         indent=4, ensure_ascii=False), file=fo_test)
        print(json.dumps(json_data[args.train_size + args.test_size:args.train_size + args.test_size + args.dev_size],
                         indent=4, ensure_ascii=False), file=fo_dev)
    return


if __name__ == '__main__':
    main()
