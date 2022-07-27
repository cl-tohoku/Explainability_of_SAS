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
        '-i', '--input', type=path.abspath, help='input dir path')
    parser.add_argument(
        '-o', '--output', type=path.abspath, help='output dir path')
    parser.add_argument(
        '-trs', '--train_size', type=int, default=350, help='train data size')
    parser.add_argument(
        '-ds', '--dev_size', type=int, default=100000000000000, help='dev data size')
    parser.add_argument(
        '-tes', '--test_size', type=int, default=100, help='train data size')
    parser.add_argument(
        '--random_seed', type=int, help='shuffle random seed', default=0)
    parser.add_argument("--seed", type=int, help="seed", default=5)
    parser.add_argument("-split", nargs="*", type=int,
                        help="split size", default=[50, 100, 200, 400, 800, 1600])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(args)
    # seedを固定
    random.seed(args.random_seed)

    # jsonデータを取得
    for path in glob(f"{args.input}/*"):
        prompt = path.split('/')[-1]
        logger.info(path)
        # train, dev, test に分割
        with open(f"{path}/{prompt}.json", 'r') as fin:
            json_data = json.load(fin)
        random.shuffle(json_data)

        train_data = json_data[:args.train_size]
        train_output_dir = f"{args.output}/{prompt}"
        os.makedirs(train_output_dir, exist_ok=True)
        dev_output_dir = f"{args.output}/{prompt}"
        os.makedirs(dev_output_dir, exist_ok=True)
        test_output_dir = f"{args.output}/{prompt}"
        os.makedirs(test_output_dir, exist_ok=True)
        with open(f"{train_output_dir}/{prompt}_train.json", 'w') as fo_train, open(f"{dev_output_dir}/{prompt}_dev.json", 'w') as fo_dev, open(f"{test_output_dir}/{prompt}_test.json", 'w') as fo_test:
            print(json.dumps(json_data[:args.train_size],
                             indent=4, ensure_ascii=False), file=fo_train)
            print(json.dumps(json_data[args.train_size:args.train_size + args.test_size],
                             indent=4, ensure_ascii=False), file=fo_test)
            print(json.dumps(json_data[args.train_size + args.test_size:args.train_size + args.test_size + args.dev_size],
                             indent=4, ensure_ascii=False), file=fo_dev)

        # train data を分割する
        for train_data_size in args.split:
            for seed in range(args.seed):
                random.seed(seed)
                random.shuffle(json_data)
                output_dir = f"{args.output}/{prompt}/data_size"
                os.makedirs(output_dir, exist_ok=True)
                output_path = f"{output_dir}/{prompt}_train.{train_data_size}.{seed}.json"
                with open(output_path, 'w') as fout:
                    print(json.dumps(
                        json_data[0:train_data_size], indent=4, ensure_ascii=False), file=fout, )
    return


if __name__ == '__main__':
    main()
