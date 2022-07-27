'''
説明
python src/map_embedding_by_umap.py -itp ~/University/Lab/SAS/results/Y14_2-2_1_4/50/0/Y14_2-2_1_4_train_attention_hidden_ref.pickle -itj ~/University/Lab/SAS/data_acl-bea19/Y14_2-2_1_4/data_size/Y14_2-2_1_4_train.50.0.json -ipp ~/University/Lab/SAS/results/Y14_2-2_1_4/50/0/Y14_2-2_1_4_test_hidden_pred_trg_dist_evidence.pickle -ipj ~/University/Lab/SAS/data_acl-bea19/Y14_2-2_1_4/Y14_2-2_1_4_test.json
python src/map_embedding_by_umap.py -itp /work01/tasuku/project/JapaneseSAS_BERT/results/Y14_2-2_1_4/50/0/Y14_2-2_1_4_train_attention_hidden_ref.pickle -itj /home/hiro819/projects/JapaneseSAS/data/japanese_sas/exp_data/data_acl-bea19/Y14_2-2_1_4/data_size/Y14_2-2_1_4_train.50.0.json
'''
import argparse
import logzero
from logzero import logger
import logging
from os import path
import pickle
import json
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import normalize
import umap
import seaborn as sns
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from getinfo import TrainInfo
# log_format = '%(color)s 【%(levelname)1.1s】%(end_color)s %(message)s'
# formatter = logzero.LogFormatter(fmt=log_format)
# logzero.formatter(formatter)
#
# logger.setLevel(logging.DEBUG)


def load_hidden_vector(pickle_file_path: str) -> list:
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
        # 次元：(採点項目，答案index，隠れ層次元)
        gold_list = list(
            map(int, data["gold"]))
        if data["pred"] is not None:
            pred_list = list(
                map(int, data["pred"]))
        else:
            pred_list = None
    return None, data["hidden_states"], gold_list, pred_list


def load_hidden_vector_for_pred(pickle_file_path: str) -> list:
    with open(pickle_file_path, 'rb') as f:
        [pred_hidden_list, pred_score_list,
         pred_gold_score_list, _, _] = pickle.load(f)
        # -1indexはスペルミスの話なので無視
        pred_hidden_list = pred_hidden_list[:-1]
        pred_score_list = list(
            map(lambda x_list: list(map(int, x_list)), pred_score_list[:-1]))
        # 最初に全体点の次元があるのでそれも削除
        pred_gold_score_list = pred_gold_score_list[1:-1].numpy().tolist()

    assert len(pred_hidden_list) == len(pred_score_list) == len(
        pred_gold_score_list), f"{len(pred_hidden_list)}, {len(pred_score_list)}, {len(pred_gold_score_list)}"

    return pred_hidden_list, pred_score_list, pred_gold_score_list,


def load_answer_detail(json_file_path: str, score_term: str) -> list:
    with open(json_file_path, 'r') as f:
        json_data_list = json.load(f)
    # 採点基準の数を計算
    score_term_num = 0
    for score_term_idx in range(10):
        if chr(65 + score_term_idx) in json_data_list[0]:
            score_term_num += 1
    answer_list = []
    score_list = []
    attention_list = []

    for json_data in json_data_list:
        # 5個の項目関係ないデータ，各項目に関して三つのデータがあるので
        answer_list.append(json_data['Char'].replace(' ', ''))
        score_list.append(json_data[f'{score_term}_Score'])
        attention_list.append(
            json_data[f'C_{score_term}'].replace(' ', ''))
    return answer_list, attention_list, score_list


def dimention_reduction_umap(tensor_data: List):
    mapper = umap.UMAP(random_state=10, metric="euclidean")
    tensor_data = mapper.fit_transform(tensor_data)
    return tensor_data


def depict_scatter(train_embedded, train_label, train_answer, pred_embedded, pred_label, pred_gold_score, pred_answer, title, save_path=None):
    train_tsne_df = pd.DataFrame(
        {"X": train_embedded[:, 0], "Y": train_embedded[:, 1], 'score': train_label, 'answer': train_answer})
    pred_tsne_df = pd.DataFrame(
        {"X": pred_embedded[:, 0], "Y": pred_embedded[:, 1], 'gold': pred_gold_score, 'pred': pred_label, 'answer': pred_answer})

    symbol = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down',
              'triangle-left', 'triangle-right', 'triangle-ne', 'triangle-se', 'triangle-sw', 'star']
    symbol2 = {"train": "circle", "pred": "cross"}
    fig = go.Figure()

    for gs in np.sort(train_tsne_df['score'].unique()):
        if gs in train_tsne_df['score']:
            train_df = train_tsne_df[train_tsne_df['score'] == gs]
            fig.add_trace(go.Scatter(x=train_df['X'], y=train_df['Y'],
                                     mode='markers',
                                     name=f'train_{gs}',
                                     marker=dict(colorscale='Viridis'),
                                     text=train_df['answer'],
                                     marker_symbol=symbol2["train"],
                                     opacity=0.5))

    for gs in np.sort(train_tsne_df['score'].unique()):
        pred_df = pred_tsne_df[pred_tsne_df['gold'] == gs]
        if not pred_df.empty:
            for p in pred_df['pred'].unique():
                p_df = pred_df[pred_df['pred'] == p]
                if not p_df.empty:
                    fig.add_trace(go.Scatter(x=p_df['X'], y=p_df['Y'],
                                             mode='markers',
                                             name=f'pred{p}_gold{gs}',
                                             text=p_df['answer'],
                                             marker_symbol=symbol2["pred"],
                                             opacity=0.5 if p == gs else 1.0))
    fig.update_layout(title=title)
    fig.update_traces(marker=dict(size=12, line=dict(
        width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
    # fig.show()
    if save_path is not None:
        fig.write_html(save_path)

    return fig


def map_embedding(input_train_pickle_file, input_pred_pickle_file, input_train_json_file, input_pred_json_file, item, save_path):

    # output_file = args.output

    train_attention_weights, train_hidden_states, train_score_references, _ = load_hidden_vector(
        input_train_pickle_file)

    train_answer_list, _, _ = load_answer_detail(
        input_train_json_file, score_term=item)

    pred_attention_weights, pred_hidden_states, pred_gold_score_list, pred_score_list = load_hidden_vector(
        input_pred_pickle_file)

    pred_answer_list, _, _ = load_answer_detail(
        input_pred_json_file, score_term=item)

    assert len(train_answer_list) == len(
        train_hidden_states), f'dimention error: {len(train_answer_list)}, {len(train_hidden_states)}'

    # ノルムを揃える
    # if args.normalize is True:
    #     for i in range(len(train_hidden_states)):
    #         train_hidden_states[i] = normalize(
    #             train_hidden_states[i], norm='l2', axis=1)
    #         pred_hidden_states[i] = normalize(
    #             pred_hidden_states[i], norm='l2', axis=1)

    # 採点項目ごとに次元削減
    # 結合
    hidden_state = np.vstack(
        [train_hidden_states, pred_hidden_states])
    # 次元圧縮
    hidden_state = dimention_reduction_umap(
        hidden_state)
    # 元に戻す
    train_hidden_state, pred_hidden_state = hidden_state[0:len(
        train_hidden_states), :], hidden_state[len(train_hidden_states):, :]

    # 結果を二次元でプロット
    fig = depict_scatter(train_embedded=train_hidden_state, train_label=train_score_references, train_answer=train_answer_list, pred_embedded=pred_hidden_state,
                         pred_label=pred_score_list, pred_gold_score=pred_gold_score_list, pred_answer=pred_answer_list, title=f"Item {item}", save_path=save_path)
    return fig


def main(args):
    logger.info(args)

    input_train_pickle_file = args.input_train_pickle_file
    input_pred_pickle_file = args.input_pred_pickle_file
    input_train_json_file = args.input_train_json_file
    input_pred_json_file = args.input_pred_json_file
    # output_file = args.output

    train_attention_weights, train_hidden_states, train_score_references, _ = load_hidden_vector(
        input_train_pickle_file)

    train_answer_list, _, _ = load_answer_detail(
        input_train_json_file, score_term=args.item)

    pred_attention_weights, pred_hidden_states, pred_gold_score_list, pred_score_list = load_hidden_vector(
        input_pred_pickle_file)

    pred_answer_list, _, _ = load_answer_detail(
        input_pred_json_file, score_term=args.item)

    assert len(train_answer_list) == len(
        train_hidden_states), f'dimention error: {len(train_answer_list)}, {len(train_hidden_states[0])}'

    # ノルムを揃える
    if args.normalize is True:
        for i in range(len(train_hidden_states)):
            train_hidden_states[i] = normalize(
                train_hidden_states[i], norm='l2', axis=1)
            pred_hidden_states[i] = normalize(
                pred_hidden_states[i], norm='l2', axis=1)

    # 採点項目ごとに次元削減
    # 結合
    hidden_state = np.vstack(
        [train_hidden_states, pred_hidden_states])
    # 次元圧縮
    hidden_state = dimention_reduction_umap(
        hidden_state)
    # 元に戻す
    train_hidden_state, pred_hidden_state = hidden_state[0:len(
        train_hidden_states), :], hidden_state[len(train_hidden_states):, :]

    # 結果を二次元でプロット
    depict_scatter(train_embedded=train_hidden_state, train_label=train_score_references, train_answer=train_answer_list, pred_embedded=pred_hidden_state,
                   pred_label=pred_score_list, pred_gold_score=pred_gold_score_list, pred_answer=pred_answer_list, title=f"Item {args.item}", save_path=args.s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-itp', '--input_train_pickle_file', type=path.abspath, help='input train pickle file path')
    # parser.add_argument(
    #     '-ipp', '--input_pred_pickle_file', type=path.abspath, help='input predict pickle file path')
    # parser.add_argument(
    #     '-itj', '--input_train_json_file', type=path.abspath, help='input train json file path')
    # parser.add_argument(
    #     '-ipj', '--input_pred_json_file', type=path.abspath, help='input predict json file path')
    parser.add_argument('--normalize', type=bool,
                        default=False, help='do normalization or not')
    # parser.add_argument("-item", dest="item", type=str,
    #                     default=False, help='item name')
    # parser.add_argument("-s", dest="s", type=str,
    #                     default=False, help='save path')

    parser.add_argument("--train_info_path", "-info", type=path.abspath)
    args = parser.parse_args()
    info = pickle.load(open(args.train_info_path, "rb"))
    args.input_train_pickle_file = f"{info.out_dir}_train_result.pickle"
    args.input_pred_pickle_file = f"{info.out_dir}_test_result.pickle"
    args.input_train_json_file = f"{info.train_dir}"
    args.input_pred_json_file = f"{info.test_dir}"
    args.item = info.item
    args.s = f"{info.out_dir}_embedding_map.html"

    main(args)
