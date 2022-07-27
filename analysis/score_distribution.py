'''
説明
'''
import sys
import os
sys.path.append(os.pardir)

import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
from glob import glob
import json
from getinfo import ranges
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from collections import defaultdict


rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio',
                               'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_json_path', type=path.abspath)
    parser.add_argument(
        '-o', '--output_dir', type=path.abspath)
    parser.add_argument('--prompt', type=str, help="prompt name")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)
    items = ['A', 'A_1', 'A_2', 'A_3', 'B', 'B_1', 'B_2', 'C', 'D', 'E']
    # for prompt_file_path in glob(f"{args.input_json_path}/*"):

    # prompt = prompt_file_path.split('/')[-1]
    # fig, ax = plt.subplots(2, 2, dpi=200)
    # hist_data = defaultdict(
    #     lambda: {'A': [], 'B': [], 'C': [], 'D': [], 'E': []})
    prompt = args.prompt
    hist_data = defaultdict(lambda: [0 for _ in range(10)])
    with open(args.input_json_path, 'r') as f:
        json_data = json.load(f)
    for data in json_data:
        for item in items:
            if prompt in ranges[item].keys():
                # hist_data[mode].append(
                #     int(data[f"{item}_Score"]))
                score = int(data[f"{item}_Score"])
                hist_data[item][score] += 1
    for item in items:
        if prompt in ranges[item].keys():
            max_score = ranges[item][prompt][-1]
            plt.bar(range(max_score + 1), hist_data[item][:max_score + 1])
            # plt.hist(hist_data[item], bins=[
            #          i-0.5 for i in range(max(hist_data[item])+1)], range=(0, max(hist_data[item])+1))
            # plt.hist(hist_data[item], bins=len(
            #     set(hist_data[item])), range=(0, max(hist_data[item])+1),)
            plt.title(f"{prompt}_{item}")
            plt.locator_params(axis='x', integer=True)
            plt.savefig(f"{args.output_dir}/{prompt}_{item}.png")
            # plt.show()
            plt.clf()
            plt.close()
    return


if __name__ == '__main__':
    main()
