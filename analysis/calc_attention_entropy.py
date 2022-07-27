"""
アテンションのエントロピーを計算して変なアテンションを貼ってないか確認する
"""

import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
import pickle
import os
import math
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input df path')
    parser.add_argument("--method", '-m', default="Attention_Weights")
    args = parser.parse_args()
    return args

def entropy(p):
    return -1 * p * math.log(p)

def calc_norm_entropy(p_list):
    return sum([entropy(p) for p in p_list]) / math.log(len(p_list))

def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)

    df = pickle.load(open(args.input,'rb'))

    featuremap = df[args.method].to_numpy().tolist()

    norm_ent_list = []
    for fm in featuremap:
        norm_ent = calc_norm_entropy(fm)
        norm_ent_list.append(norm_ent)
    ave_norm_ent = np.mean(norm_ent_list)

    *_, prompt, item, train_size, attn_size, seed = os.path.split(args.input)[0].split('/')
    print(prompt, item, train_size, attn_size, seed, args.method,ave_norm_ent ,sep='\t')

    return

if __name__ == '__main__':
    main()
