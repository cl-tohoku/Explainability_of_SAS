'''
2つの手法間のmseを計算する
'''
from scipy.stats import kendalltau
import numpy as np
import argparse
import pickle
from os import path
import os
from sklearn.metrics import mean_squared_error


def calc_mse(X,Y, is_mean=True):
    # print(X,Y)
    assert len(X) == len(Y)
    mse_list = []
    for x,y in zip(X,Y):
        mse = mean_squared_error(x,y)
        mse_list.append(mse)
    if is_mean is True:
        return np.mean(mse_list)
    else:
        return mse_list


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

def _df_to_features(df_path, target, debug):
    df = pickle.load(open(df_path,'rb'))
    features = df[target]
    if debug:
        features = features[:10]
    return features

def main():
    args = parse_args()
    *_, prompt, item, train_size, attn_size, seed = os.path.split(args.fir_input)[0].split('/')

    fir_features = _df_to_features(args.fir_input, args.fir_target, args.debug)
    sec_features = _df_to_features(args.sec_input, args.sec_target, args.debug)

    mse = calc_mse(fir_features, sec_features)
    print(prompt, item, train_size, attn_size, seed, args.fir_target, args.sec_target,mse,sep='\t')

if __name__ == "__main__":
    main()