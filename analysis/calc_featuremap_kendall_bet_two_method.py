'''
2つの手法間のmseを計算する
'''
from scipy.stats import kendalltau
import numpy as np
import argparse
import pickle
from os import path
import os

def calc_kendall(X,Y, is_mean=True):
    # print(X,Y)
    assert len(X) == len(Y)
    correlation_list = []
    pvalue_list = []
    for x,y in zip(X,Y):
        c, p = kendalltau(x,y)
        correlation_list.append(c)
        pvalue_list.append(p)
    if is_mean is True:
        # print(correlation_list)
        # print(pvalue_list)
        return np.mean(correlation_list), np.mean(pvalue_list)
    else:
        return correlation_list, pvalue_list

def feature_to_rank(feature_map):
    ret = []
    for f in feature_map:
        ret.append(np.argsort(np.array(f)*-1).tolist())
    return ret

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fir_input', type=path.abspath, help='input file path')
    parser.add_argument(
        '--sec_input', type=path.abspath, help='input file path')
    parser.add_argument("--fir_target",type=str, default="Attention_Weights",choices=["Attention_Weights","Saliency", "SmoothGrads","Integrated_Gradients"])
    parser.add_argument("--sec_target",type=str, default="Attention_Weights",choices=["Attention_Weights","Saliency", "SmoothGrads","Integrated_Gradients"])
    # parser.add_argument(
    #     '-o', '--output', type=path.abspath, help='output file path')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args

def _df_to_rank(df_path, target, debug):
    df = pickle.load(open(df_path,'rb'))
    features = df[target]
    if debug:
        features = features[:10]
    ranks = feature_to_rank(features)
    return ranks

def main():
    args = parse_args()
    *_, prompt, item, train_size, attn_size, seed = os.path.split(args.fir_input)[0].split('/')
    fir_ranks = _df_to_rank(args.fir_input, args.fir_target, args.debug)
    sec_ranks = _df_to_rank(args.sec_input, args.sec_target, args.debug)

    correlation, pvalue = calc_kendall(fir_ranks,sec_ranks)
    print(prompt, item, train_size, attn_size, seed, args.fir_target, args.sec_target,correlation, pvalue,sep='\t')

if __name__ == "__main__":
    main()