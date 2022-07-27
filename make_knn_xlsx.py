'''
説明
舟山実装のコード(https://github.com/cl-tohoku/JapaneseSAS_BERT/blob/236c9ff153142abc402c77d753288cdb4f4fe979/train_instance_base_item_scoring_model.py)
で出力されたデータをxlsxデータにして保存するコード
'''
import argparse
import logzero
from logzero import logger
from os import path
import logging

import pickle
import json

from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import umap
from typing import List
import copy


import xlsxwriter

log_format = '%(color)s[%(levelname)1.1s]%(end_color)s %(message)s'
formatter = logzero.LogFormatter(fmt=log_format)
logzero.formatter(formatter)
logzero.loglevel(logging.INFO)


def is_same_score_bet_json_and_pickle(json_data, score_data):
    for i in range(len(score_data)):
        score_term = chr(65 + i)
        for j in range(len(score_data[i])):
            assert json_data[j][f"{score_term}_Score"] == score_data[i][j], f"score don't match {i},{j}"
    return


def write_row_data_to_xlsx(worksheet, column_index_dict, column_data_dict, row):
    assert set(column_data_dict.keys()) == set(column_index_dict.keys(
    )), f"key error\n{' '.join(column_index_dict.keys())}\n{' '.join(column_data_dict.keys())}"
    for key, column_name in column_index_dict.items():
        value = column_data_dict[key]
        try:
            worksheet.write(f"{column_name}{row}", *value)
        except:
            flag = worksheet.write_rich_string(f"{column_name}{row}", *value)
            if flag == -5:
                print(*value)


def make_column_index(index: int) -> str:
    '''
    エクセルの列指定の文字列を生成
    27列目→AA列
    '''
    ret = []
    Z_index = ord('Z')
    A_index = ord('A')
    letter_num = ord('Z') - ord('A') + 1
    d = index + 1
    while 0 != d:
        d -= 1
        d, m = divmod(d, letter_num)
        ret.append(m)
    ret = ''.join(map(lambda c: chr(c + A_index), ret[::-1]))
    return ret


def load_pickle(pickle_file_path: str):
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    return data["hidden_states"], data["gold"].numpy(), data["pred"], data['distance'], data['evidence'], data["attentions"]


# def load_pickle_for_pred(pickle_file_path: str):
#     with open(pickle_file_path, 'rb') as f:
#         data = pickle.load(pickle_file_path)
#         hidden_vector = data["hidden_states"]
#     return prep_hidden_list_list, pred_score_list, pred_gold_score_list,


def dimention_reduction_umap(train_hid: List, pred_hid: List, dim: int) -> List:
    '''
    次元圧縮
    '''
    train_ret = []
    pred_ret = []
    assert len(train_hid) == len(pred_hid), 'error'
    for i in range(len(train_hid)):
        concat = np.vstack(
            [train_hid[i], pred_hid[i]])
        mapper = umap.UMAP(random_state=0, n_components=dim)
        concat = mapper.fit_transform(concat)
        train_ret.append(concat[0:len(train_hid[i]), :])
        pred_ret.append(concat[len(train_hid[i]):, :])
    assert len(train_hid) == len(train_ret), 'error'
    train_hid = np.array(train_hid)
    pred_hid = np.array(pred_hid)
    return train_ret, pred_ret


def calc_trust_score(knn_model, all_distance, hidden_state, pred_score):
    dist_wo_all_distance = 1e9
    for point in knn_model.keys():
        if pred_score != point:
            distance, _ = knn_model[point].kneighbors(
                [hidden_state], n_neighbors=1)
            dist_wo_all_distance = min(dist_wo_all_distance, distance[0][0])
    trust_score = (dist_wo_all_distance) / \
        (all_distance + dist_wo_all_distance)
    return trust_score


def create_work_sheet(workbook, sheet_name):
    worksheet = workbook.add_worksheet(sheet_name)
    column_list = ["pred_id", "knn", "train_id", "score_term", "answer_sentence",
                   "gold_score", "pred_score", 'distance', "trust_score", "norm", ]
    column_index_dict = dict([(column, make_column_index(i))
                              for i, column in enumerate(column_list)])
    column_data_dict = {"pred_id": ['predId'], "knn": ['knn'], "train_id": ['trainId'], "score_term": ['採点項目'], "answer_sentence": ['答案'],
                        "gold_score": ['正解'], "pred_score": ['予測'], 'distance': ['距離'], "trust_score": ["確信度"], 'norm': ["ノルム"], }
    worksheet.set_column("A:A", 4)
    worksheet.set_column("B:B", 4)
    worksheet.set_column("C:C", 4)
    worksheet.set_column("D:D", 5)
    worksheet.set_column("E:E", 130)
    worksheet.set_column("F:F", 5)
    worksheet.set_column("G:G", 5)
    worksheet.set_column("H:H", 5)
    worksheet.set_column("I:I", 5)
    worksheet.set_column("J:J", 5)
    # worksheet.set_column("K:K", 5)
    write_row_data_to_xlsx(worksheet=worksheet, column_data_dict=column_data_dict,
                           column_index_dict=column_index_dict, row=1)
    return worksheet


def _deepcopy(dictionary):
    ret = dict()
    for k, v in dictionary.items():
        if k == "answer_sentence":
            continue
        ret[k] = copy.deepcopy(v)
    return ret


def main(args):
    # logzero.logfile(args.log)  # 追加: logfileの作成
    # logger.info(args)

    train_pickle_file = args.train_pickle_file
    pred_pickle_file = args.predict_pickle_file
    pred_json_file = args.predict_json_file
    gold_train_json_file = args.gold_train_json_file
    gold_pred_json_file = args.gold_pred_json_file
    output = args.output
    correct_n = args.correct_n
    miss_n = args.miss_n
    score_term = args.score_term

    if args.char:
        attention_key = f"C_{score_term}"
        answer_key = "Char"
    else:
        attention_key = f"{score_term}"
        answer_key = "mecab"

    column_list = ["pred_id", "knn", "train_id", "score_term", "answer_sentence",
                   "gold_score", "pred_score", 'distance', "trust_score", "norm"]
    column_index_dict = dict([(column, make_column_index(i))
                              for i, column in enumerate(column_list)])

    workbook = xlsxwriter.Workbook(output)  # 新規xlsxファイル作成
    # sheetにheaderを書く
    miss_worksheet = create_work_sheet(workbook, sheet_name='miss')
    correct_worksheet = create_work_sheet(workbook, sheet_name='correct')

    # スプレッドシートの列をincrementしながらデータを追加
    miss_row_index = 2
    correct_row_index = 2

  # identification cueに色をつけるもの
    # color = dict()
    # for i, c in enumerate(['red', 'blue', 'green', 'orange', 'gray']):
    #     color[chr(ord('A')+i)
    #           ] = workbook.add_format({'color': c, "bold": True})
    color = workbook.add_format({'color': 'blue', "bold": True})

    # 訓練データのアテンションと隠れ層ベクトルを取得する
    t_hidden_list, t_gold_score_list, _, _, _, t_attentions = load_pickle(
        train_pickle_file)

    # 評価データのの隠れ層ベクトルを取得する
    p_hidden_list, p_gold_score_list, p_pred_score_list, *_ = load_pickle(
        pred_pickle_file)

    # 訓練データと予測データのjsonファイルを取得
    with open(gold_train_json_file, 'r') as f:
        gold_train_json = json.load(f)
    with open(gold_pred_json_file, 'r') as f:
        gold_pred_json = json.load(f)
    if pred_json_file is not None:
        with open(pred_json_file, 'r') as f:
            pred_json = json.load(f)

    gold_score_idx_dict_list = defaultdict(
        list)  # key:得点 value:keyの得点の答案のindex
    logger.info(f"===={score_term}====")
    for index, gs in enumerate(t_gold_score_list):
        gold_score_idx_dict_list[gs].append(index)  # 得点がgsである答案ごとに分ける

    logger.debug("--point | count--")
    for gs in gold_score_idx_dict_list.keys():
        logger.debug(
            f"{gs} | {len(gold_score_idx_dict_list[gs])}")

    neigh_model = dict()
    # 各設問ごとのknnモデルを作成
    for gs in gold_score_idx_dict_list.keys():
        neigh_model[gs] = KNeighborsClassifier(n_neighbors=correct_n)
        neigh_model[gs].fit(t_hidden_list
                            [gold_score_idx_dict_list[gs]], np.array(t_gold_score_list)[gold_score_idx_dict_list[gs]])
    # 全ての答案からknnするモデルを作成
    all_neigh_model = KNeighborsClassifier(n_neighbors=miss_n)
    all_neigh_model.fit(
        t_hidden_list, t_gold_score_list)

    logger.debug("--pred | gold--")

    # p_pred_score_list = all_neigh_model.predict(
    #     p_hidden_list).reshape(-1)  # 実際に計算する場合はこっち使う

    for pred_idx, (pred, gold) in enumerate(zip(p_pred_score_list, p_gold_score_list)):
        # 全体と答案間での最近傍n個を取得
        all_knn_distance_list, all_knn_index_list = all_neigh_model.kneighbors(
            [p_hidden_list[pred_idx]])
        all_knn_distance_list = all_knn_distance_list.reshape(-1).tolist()
        all_knn_index_list = all_knn_index_list.reshape(-1).tolist()

        # trust scoreを計算
        trust_score = calc_trust_score(
            neigh_model, all_knn_distance_list[0], p_hidden_list[pred_idx], pred)

        # l2ノルムを計算
        norm = np.linalg.norm(p_hidden_list[pred_idx], ord=2)

        gold_index = None
        gold_distance = None
        # 訓練事例内にgoldと同じ得点の答案が存在する場合
        if gold in gold_score_idx_dict_list.keys():
            neign_num = min(correct_n, len(
                gold_score_idx_dict_list[gold]))
            # 答案のgoldとの距離を取得
            gold_distance, gold_index = neigh_model[gold].kneighbors(
                [p_hidden_list[pred_idx]], n_neighbors=neign_num)
            gold_distance = gold_distance[0][0]

            # ここでのindexはあくまでindex_dict_listのインデックスを指しているので，jsonのindexを刺す様に変更
            gold_index = gold_score_idx_dict_list[gold][gold_index[0][0]]

        # 採点対象の答案を出力
        column_data_dict = {"pred_id": [pred_idx], "knn": [-1], "train_id": [-1], "score_term": [score_term], "answer_sentence": [], "gold_score": [gold_pred_json[pred_idx][f'{score_term}_Score']],
                            "pred_score": [gold_train_json[all_knn_index_list[0]][f'{score_term}_Score']], 'distance': [None], 'trust_score': [trust_score], "norm": [norm], }
        attention = gold_pred_json[pred_idx][attention_key].split()
        answer = gold_pred_json[pred_idx][answer_key].split(' ')
        for att, char in zip(attention, answer):
            if att == '1':
                column_data_dict['answer_sentence'].append(
                    color)
            column_data_dict['answer_sentence'].append(char)
        # ここを入れるとエラーが消える
        column_data_dict['answer_sentence'].append(
            workbook.add_format({"bold": False}))
        column_data_dict['answer_sentence'].append(" ")

        # predのアテンションを表示したいとき
        if pred_json is not None:
            # column_data_dict_for_pred_attention = copy.deepcopy(
            #     column_data_dict)
            column_data_dict_for_pred_attention = _deepcopy(column_data_dict)
            column_data_dict_for_pred_attention['answer_sentence'] = []
            attention = list(
                map(float, pred_json[pred_idx][f'{score_term}_attention'][1:-1].split()))
            answer = gold_pred_json[pred_idx][answer_key].split(' ')
            for att, char in zip(attention, answer):
                att_color = f'{255-min(255,int(att*255)*5):0>2X}'
                att_color = f"#{att_color}{att_color}{att_color}"
                column_data_dict_for_pred_attention['answer_sentence'].append(
                    workbook.add_format({'color': att_color, "bold": True}))
                column_data_dict_for_pred_attention["answer_sentence"].append(
                    char)
            # ここを入れるとエラーが消える？
            column_data_dict_for_pred_attention['answer_sentence'].append(
                workbook.add_format({"bold": False}))
            column_data_dict_for_pred_attention['answer_sentence'].append(" ")

        if pred != gold:
            logger.debug(
                f"{pred} | {gold}")
            write_row_data_to_xlsx(
                worksheet=miss_worksheet, column_data_dict=column_data_dict_for_pred_attention, column_index_dict=column_index_dict, row=miss_row_index)
            miss_row_index += 1
            write_row_data_to_xlsx(
                worksheet=miss_worksheet, column_data_dict=column_data_dict, column_index_dict=column_index_dict, row=miss_row_index)
            miss_row_index += 1
        else:
            write_row_data_to_xlsx(
                worksheet=correct_worksheet, column_data_dict=column_data_dict_for_pred_attention, column_index_dict=column_index_dict, row=correct_row_index)
            correct_row_index += 1

            # ここがエラーの原因らしい
            write_row_data_to_xlsx(
                worksheet=correct_worksheet, column_data_dict=column_data_dict, column_index_dict=column_index_dict, row=correct_row_index)
            correct_row_index += 1

        if pred != gold:
            # 正解答案を出力
            norm = np.linalg.norm(
                t_hidden_list[gold_index], ord=2) if gold_index else None
            # cos_sim = cosine_similarity([t_hidden_list[gold_index]], [
            #                             p_hidden_list[pred_idx]])[0] if gold_index else None
            cos_sim = None
            column_data_dict = {"pred_id": [pred_idx], "knn": [0], "train_id": [gold_index], "score_term": [score_term], "answer_sentence": [],
                                "gold_score": [gold_pred_json[pred_idx][f'{score_term}_Score']], "pred_score": [None], 'distance': [gold_distance], 'trust_score': [trust_score], "norm": [norm], }
            if gold_index is not None:
                attention = gold_train_json[gold_index][attention_key].split(
                )
                for char_idx, char in enumerate(gold_train_json[gold_index][answer_key].split(' ')):
                    if attention[char_idx] == '1':
                        column_data_dict['answer_sentence'].append(
                            color)
                    column_data_dict['answer_sentence'].append(char)
            else:
                column_data_dict['answer_sentence'].append(' ')
            column_data_dict['answer_sentence'].append(
                workbook.add_format({'color': att_color, "bold": True}))
            write_row_data_to_xlsx(worksheet=miss_worksheet, column_data_dict=column_data_dict,
                                   column_index_dict=column_index_dict, row=miss_row_index)
            miss_row_index += 1
        # else:
            # 最も近い正解じゃない答案を出力？？？予定

        for i_m in range(len(all_knn_index_list)):
            column_data_dict = {"pred_id": [pred_idx], "knn": [
                i_m + 1], "train_id": [all_knn_index_list[i_m]], 'trust_score': [trust_score], }

            column_data_dict["score_term"] = [score_term]
            column_data_dict['answer_sentence'] = []
            attention = gold_train_json[all_knn_index_list[i_m]][attention_key].split(
            )
            answer = gold_train_json[all_knn_index_list[i_m]][answer_key].split(
                ' ')
            for att, char in zip(attention, answer):
                if att == '1':
                    column_data_dict['answer_sentence'].append(
                        color)
                column_data_dict['answer_sentence'].append(char)
            # ここを入れるとエラーが消える
            column_data_dict['answer_sentence'].append(
                workbook.add_format({'color': att_color, "bold": True}))
            column_data_dict['answer_sentence'].append(' ')

            # 間違った近傍事例の得点を出力
            column_data_dict['pred_score'] = [
                gold_train_json[all_knn_index_list[i_m]][f'{score_term}_Score']]
            column_data_dict['gold_score'] = [None]

            # 間違った近傍事例の距離
            column_data_dict['distance'] = [all_knn_distance_list[i_m]]

            # l2ノルムを計算
            column_data_dict['norm'] = [np.linalg.norm(
                t_hidden_list[all_knn_index_list[i_m]], ord=2)]

            # コサイン類似度計算
            # column_data_dict['cos_sim'] = cosine_similarity([t_hidden_list[all_knn_index_list[i_m]]], [
            #     p_hidden_list[pred_idx]])[0]

            # if pred != gold:
            #     logger.debug(
            #         f"{pred} | {gold}")
            #     write_row_data_to_xlsx(
            #         worksheet=miss_worksheet, column_data_dict=column_data_dict, column_index_dict=column_index_dict, row=miss_row_index)
            #     miss_row_index += 1
            # else:
            #     write_row_data_to_xlsx(
            #         worksheet=correct_worksheet, column_data_dict=column_data_dict, column_index_dict=column_index_dict, row=correct_row_index)
            #     correct_row_index += 1

#############新たに追加：train dataの方もアテンションを表示させたい#######################
            # predのアテンションを表示したいとき
            if t_attentions is not None:
                # column_data_dict_for_pred_attention = copy.deepcopy(
                #     column_data_dict)
                column_data_dict_for_pred_attention = _deepcopy(
                    column_data_dict)
                column_data_dict_for_pred_attention['answer_sentence'] = []
                attention = t_attentions[all_knn_index_list[i_m]].tolist()
                for att, char in zip(attention, answer):
                    att_color = f'{255-min(255,int(att*255)*5):0>2X}'
                    att_color = f"#{att_color}{att_color}{att_color}"
                    column_data_dict_for_pred_attention['answer_sentence'].append(
                        workbook.add_format({'color': att_color, "bold": True}))
                    column_data_dict_for_pred_attention["answer_sentence"].append(
                        char)
                # ここを入れるとエラーが消える？
                column_data_dict_for_pred_attention['answer_sentence'].append(
                    workbook.add_format({"bold": False}))
                column_data_dict_for_pred_attention['answer_sentence'].append(
                    " ")

            if pred != gold:
                write_row_data_to_xlsx(
                    worksheet=miss_worksheet, column_data_dict=column_data_dict_for_pred_attention, column_index_dict=column_index_dict, row=miss_row_index)
                miss_row_index += 1
                logger.debug(
                    f"{pred} | {gold}")
                write_row_data_to_xlsx(
                    worksheet=miss_worksheet, column_data_dict=column_data_dict, column_index_dict=column_index_dict, row=miss_row_index)
                miss_row_index += 1
            else:
                write_row_data_to_xlsx(
                    worksheet=correct_worksheet, column_data_dict=column_data_dict_for_pred_attention, column_index_dict=column_index_dict, row=correct_row_index)
                correct_row_index += 1

                write_row_data_to_xlsx(
                    worksheet=correct_worksheet, column_data_dict=column_data_dict, column_index_dict=column_index_dict, row=correct_row_index)
                correct_row_index += 1
###################################

    workbook.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-pj", "--predict_json_file", type=path.abspath, default=None, help="predict json file path"
    # )
    # parser.add_argument(
    #     "-tp", "--train_pickle_file", type=path.abspath, help="trapickle file path"
    # )
    # parser.add_argument(
    #     "-pp", "--predict_pickle_file", type=path.abspath, help="predict pickle file path"
    # )
    # parser.add_argument(
    #     '-gt', '--gold_train_json_file', type=path.abspath, help='gold train json file path')
    # parser.add_argument(
    #     '-gp', '--gold_pred_json_file', type=path.abspath, help='gold pred json file path')
    # parser.add_argument(
    #     '-o', '--output', type=path.abspath, help='output file path')
    parser.add_argument('--log',
                        default=None, help='Path to log file')
    parser.add_argument('-cn', '--correct_n',
                        type=int, default=1, help='knn of correct')
    parser.add_argument('-mn', '--miss_n',
                        type=int, default=3, help='knn of miss')
    # parser.add_argument('-st', '--score_term',
    #                     type=str, required=True, help="score term")
    parser.add_argument("--char", "-char", dest="char", default=False,
                        action='store_true', help="use char attention")
    parser.add_argument("--train_info_path", "-info", type=path.abspath)
    args = parser.parse_args()
    info = pickle.load(open(args.train_info_path, "rb"))
    args.train_pickle_file = f"{info.out_dir}_train_result.pickle"
    args.predict_pickle_file = f"{info.out_dir}_test_result.pickle"
    args.gold_train_json_file = f"{info.train_dir}"
    args.gold_pred_json_file = f"{info.test_dir}"
    args.predict_json_file = f"{info.out_dir}_test.json"
    args.output = f"{info.out_dir}_instance_knn.xlsx"
    args.score_term = info.item

    args.item = info.item

    main(args)
