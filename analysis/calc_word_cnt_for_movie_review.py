'''
movie review の文章中の単語数を数える
'''


import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
from glob import glob
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input data dir path')
#    parser.add_argument('--log',
#                        default="log.txt", help='Path to log file')
    args = parser.parse_args()
    return args


def get_cnt(path):
    with open(path, 'r') as f:
        sentence = ' '.join(map(lambda x: x.rstrip("\r\n"), f.readlines()))
    return len(sentence.split())


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)

    # word_cnt_dict = defaultdict(int)

    for path in glob(f"{args.input}/*.txt"):
        # word_cnt_dict[get_cnt(path)] += 1
        print(get_cnt(path))

    # for k, v in sorted(word_cnt_dict.items()):
    #     print(k, v, sep='\t')


if __name__ == '__main__':
    main()
