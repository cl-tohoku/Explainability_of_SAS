
'''
usage:
python print_eraser_result_for_excel.py -dir /work01/tasuku/project/JapaneseSAS_BERT/results/Yosemi_model_change/lstm_1_bidirectional |cut -f 5,6 --output-delimiter , |python boxplot.py --title hoge --save_path ~/bi.png

'''

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
from os import path
import pickle

import argparse


class BoxPlot:
    def __init__(self):
        return

    def plot(self, data, labels, title, save_path):
        '''
        '''
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.boxplot(data, labels=labels, showmeans=True)
        plt.savefig(save_path, dpi=200)

    def load_from_csv(self, f, sep, save_path='./boxplot.png', is_title=True, title="No title", is_header=True):
        datasets = defaultdict(list)
        title = f.readline().rstrip() if is_title else title
        if is_header:
            _ = f.readline().rstrip().split(sep)
        for line in f:
            l, d = line.rstrip().split(sep)
            if np.isnan(float(d)):
                print(d)
                continue

            datasets[l].append(float(d))
        self.plot(datasets.values(), datasets.keys(), title, save_path)

    def load_from_df(self, df, save_path='./boxplot.png', is_title=True, title="No title"):
        datasets = defaultdict(list)
        labels = set()
        for _, row in df[["flip_most", "name"]].iterrows():
            l = row["name"]
            labels.add(l)
            d = row["flip_most"]
            if 0 < d:
                datasets[l].append(float(d))
        labels = sorted(list(labels))
        self.plot(datasets.values(), labels, title, save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--title', type=str, default="No title", help='title')
    parser.add_argument("--is_title", action="store_true",
                        default=False, help="graph title")
    parser.add_argument("--save_path", type=str, default='./')
    parser.add_argument("-df", "--data_frame",
                        type=path.abspath, default=None, help='data frame')
    parser.add_argument("-sep", type=str, default=",", help='sep token')
    args = parser.parse_args()
    return args


def main(fi):
    args = parse_args()
    boxplot = BoxPlot()
    if args.data_frame is None:
        boxplot.load_from_csv(fi, title=args.title,
                              is_title=args.is_title, save_path=args.save_path, sep=args.sep)
    else:
        df = pickle.load(open(args.data_frame, 'rb'))
        boxplot.load_from_df(df, title=args.title,
                             is_title=args.is_title, save_path=args.save_path)


if __name__ == "__main__":
    main(sys.stdin)
