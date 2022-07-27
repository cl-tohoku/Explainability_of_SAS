'''
sanity pickleをjson形式に変換
'''
import argparse
import logzero
from logzero import logger
import logging
from collections import defaultdict
from os import path

import pickle
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input file path')
    parser.add_argument(
        '-o', '--output', type=path.abspath, help='output file path', default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)
    output = args.output if args.output is not None else args.input.replace(
        "training.pickle", "performance.json")

    df = pickle.load(open(args.input, 'rb'))
    data = defaultdict(dict)

    data["test_qwk"] = df.iloc[-1]["acc"]
    data["test_mse"] = [0, 0]

    json.dump(data, open(output, 'w'), indent=2)
    # print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
