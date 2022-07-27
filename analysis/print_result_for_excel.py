'''
実験結果をエクセル形式で出力
設問を指定→各設問ごとの性能をシードごとに出力
指定なし　→　シードごとの性能を平均した結果を出力
'''
import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List, Dict
from glob import glob
from collections import defaultdict
import json
import numpy as np
from domain.prompt import prompts

logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dir', type=path.abspath, help='input file dir')
    parser.add_argument(
        '-prompt', type=str, default=None)
    parser.add_argument("--include_zero_score", "-izs", dest="include_zero_score", default=False, action='store_true',
                        help="By using this, zero score answer is taken into account in the justification identification")
    args = parser.parse_args()
    return args


def print_result(dir_path: str, include_zero_score: bool):
    print("prompt\ttrain size\tattn size\tseed\tjusti\tqwk\tmse\tf1\trecall\tprecision\tauprc\tcomprehensiveness\tsufficiency")
    for prompt in get_prompt_list(dir_path):
        for item in get_item_list(dir_path, prompt):
            for train_size in get_train_size_list(dir_path, prompt, item):
                for attention_size in get_attention_size_list(dir_path, prompt, item, train_size):
                    for seed in get_seed_list(dir_path, prompt, item, train_size, attention_size):
                        prefix_file_path = f"{dir_path}/{prompt}/{item}/{train_size}/{attention_size}/{seed}/{prompt}_{item}_{train_size}_{attention_size}_{seed}"

                        # for justification_method in f_p_data.keys():
                        for justification_method in ["attn"]:
                            try:
                                evaluation_result_file_path = f"{prefix_file_path}_evaluation_result.json"

                                if path.isfile(evaluation_result_file_path):
                                    # print(f"{prompt}_{item}", end='\t')
                                    print(f"{prompts[prompt].type}_{item}}", end='\t')
                                    print(train_size, end='\t')
                                    print(attention_size, end='\t')
                                    print(seed, end='\t')
                                    print(justification_method, end='\t')

                                    with open(evaluation_result_file_path, 'r') as f:
                                        json_data = json.load(f)
                                    print(json_data["test_qwk"], end='\t')
                                    print(json_data["test_mse"][0], end='\t')
                                else:
                                    continue

                                f_p_file_path = f"{prefix_file_path}_faithfulness_and_plausibility_with_zero_score_result.json" if include_zero_score else f"{prefix_file_path}_faithfulness_and_plausibility_wo_zero_score_result.json"
                                if path.isfile(f_p_file_path):
                                    f_p_data = json.load(
                                        open(f_p_file_path, 'r'))
                                    print(f_p_data[justification_method]
                                          ["test"]["f1"], end='\t')
                                    print(f_p_data[justification_method]
                                          ["test"]["recall"], end='\t')
                                    print(f_p_data[justification_method]
                                          ["test"]["precision"], end='\t')
                                    print(f_p_data[justification_method]
                                          ["test"]["auprc"], end='\t')
                                    print(f_p_data[justification_method]
                                          ["test"]["comprehensiveness"], end='\t')
                                    print(f_p_data[justification_method]
                                          ["test"]["sufficiency"], end='\t')
                                print()
                            except:
                                print()


# def get_result(dir_path: str, include_zero_score: bool, include_norm_for_justification: bool, item_list: List[str], train_size_list: List[int], attention_size_list: List[int], ) -> Dict:
def get_result(dir_path: str, include_zero_score: bool, ) -> Dict:
    # result = { prompt:{item:{train_size:{attention_size:{seed:{}}}}}}
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))))

    for prompt in get_prompt_list(dir_path):
        for item in get_item_list(dir_path, prompt):
            for train_size in get_train_size_list(dir_path, prompt, item):
                for attention_size in get_attention_size_list(dir_path, prompt, item, train_size):
                    for seed in get_seed_list(dir_path, prompt, item, train_size, attention_size):
                        prefix_file_path = f"{dir_path}/{prompt}/{item}/{train_size}/{attention_size}/{seed}/{prompt}_{item}_{train_size}_{attention_size}_{seed}"
                        evaluation_result_file_path = f"{prefix_file_path}_evaluation_result.json"
                        if not path.isfile(evaluation_result_file_path):
                            continue
                        with open(evaluation_result_file_path, 'r') as f:
                            json_data = json.load(f)
                        result[prompt][item][train_size][attention_size][seed]['qwk'] = json_data["test_qwk"]
                        result[prompt][item][train_size][attention_size][seed]['rmse'] = json_data["test_mse"][1]

                        # # justification identificationの結果を取得
                        # _include_norm = "_include_norm" if include_norm_for_justification else ""
                        # _wo_zero_score = "_wo_zero_score" if not include_zero_score else ""
                        # file_path = f"{prefix_file_path}_attention_identification{_wo_zero_score}{_include_norm}_result.json"
                        # with open(file_path, 'r') as f:
                        #     json_data = json.load(f)
                        # result[prompt][item][train_size][attention_size][seed]["f1"] = json_data['test']["f1"]
                        # result[prompt][item][train_size][attention_size][seed]["precision"] = json_data['test']["precision"]
                        # result[prompt][item][train_size][attention_size][seed]["recall"] = json_data['test']["recall"]
                        # result[prompt][item][train_size][attention_size][seed]["auprc"] = json_data['test']["auprc"]

                        # # faithfulnessの結果を取得
                        # file_path = f"{prefix_file_path}_faithfulness.json"
                        # with open(file_path, 'r')as f:
                        #     json_data = json.load(f)
                        # result[prompt][item][train_size][attention_size][seed]["comprehensiveness"] = json_data['test']["comprehensiveness"]
                        # result[prompt][item][train_size][attention_size][seed]["sufficiency"] = json_data['test']["sufficiency"]

                        file_path = f"{prefix_file_path}_faithfulness_and_plausibility_with_zero_score_result.json" if include_zero_score else f"{prefix_file_path}_faithfulness_and_plausibility_without_zero_score_result.json"
                        json_data = json.load(open(file_path, 'r'))
                        for justification_method in json_data.keys():
                            result[prompt][item][train_size][attention_size][seed][justification_method] = json_data[justification_method]
    return result


# def get_ave_result(result):
#     ave_result = defaultdict(lambda: defaultdict(
#         lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
#     for metric in result.keys():
#         for prompt in result.keys():
#             for item in result[prompt].keys():
#                 for train_size in result[prompt][item].keys():
#                     for column_index in result[prompt][item][train_size][0].keys():
#                         tmp = []
#                         for seed in result[prompt][item][train_size].keys():
#                             tmp.append(result[prompt][item]
#                                        [train_size][seed][column_index])
#                         ave_result[prompt][item][train_size][column_index] = np.mean(
#                             tmp)
#     return ave_result


def get_seed_ave_result(result):
    ave_result = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
    for prompt in result.keys():
        for item in result[prompt].keys():
            for train_size in result[prompt][item].keys():
                for attention_size in result[prompt][item][train_size].keys():
                    if not result[prompt][item][train_size][attention_size]:
                        continue
                    # for column_index in result[prompt][item][train_size][attention_size][0].keys():
                    for column_index in result[prompt][item][train_size][attention_size][list(result[prompt][item][train_size][attention_size].keys())[0]].keys():
                        tmp = []
                        for seed in result[prompt][item][train_size][attention_size].keys():
                            tmp.append(result[prompt][item]
                                       [train_size][attention_size][seed][column_index])
                            ave_result[prompt][item][train_size][attention_size][column_index] = np.mean(
                                tmp)
    return ave_result


def print_per_seed(result: Dict, dir_path: str, prompt: str, item: str, train_size: int, attention_size: int) -> None:
    columns_title_list = get_column_list()
    # フォーマットを作る
    # 題名
    print(f"{prompt}\t{item}\ttrain_size:{train_size}\n")
    print('\t'.join(columns_title_list))
    max_seed = 0

    # シードごとの結果を取得
    for seed in get_seed_list(dir_path, prompt, item, train_size, attention_size):
        print(seed, end='\t')
        for column_index in columns_title_list[1:]:
            if seed in result[prompt][item][train_size][attention_size]:
                print(result[prompt][item][train_size][attention_size]
                      [seed][column_index], end='\t')
            else:
                print(end='\t')
        print()

    print('ave', end='\t')
    for column_index in columns_title_list[1:]:
        tmp = []
        for seed in get_seed_list(dir_path, prompt, item, train_size, attention_size):
            tmp.append(result[prompt][item][train_size]
                       [attention_size][seed][column_index])
        print(np.mean(tmp), end='\t')
    print()
    return


def print_per_prompt_item(ave_result, train_size, attention_size):
    # フォーマットを作る
    # 題名
    columns_title_list = get_column_list()
    print(f"train_size:{train_size}")
    print('\t'.join(columns_title_list))
    for prompt in ave_result.keys():
        for item in ave_result[prompt].keys():
            print(f"{prompt}_{item}", end='\t')
            for column_index in columns_title_list[1:]:
                print(ave_result[prompt][item][train_size][attention_size]
                      [column_index], end='\t')
            print()
    print("\n")
    return


def get_prompt_list(dir_path: str):
    return [path.split('/')[-1] for path in glob(f"{dir_path}/*")]


def get_item_list(dir_path: str, prompt: str):
    return [path.split('/')[-1] for path in glob(f"{dir_path}/{prompt}/*")]


def get_train_size_list(dir_path: str, prompt: str, item: str):
    return [int(path.split('/')[-1]) for path in glob(f"{dir_path}/{prompt}/{item}/*")]


def get_attention_size_list(dir_path: str, prompt: str, item: str, train_size: int):
    return [int(path.split('/')[-1]) for path in glob(f"{dir_path}/{prompt}/{item}/{train_size}/*")]


def get_seed_list(dir_path: str, prompt: str, item: str, train_size: int, attention_size: int):
    return [int(path.split('/')[-1])for path in glob(f"{dir_path}/{prompt}/{item}/{train_size}/{attention_size}/*")]


def get_column_list():
    return ["seed", "qwk", "rmse", "f1", "precision", "recall", "threshold", 'auprc', "comprehensiveness", "sufficiency"]


def print_per_train_size(ave_result, metric: str):
    columns_title_list = ["prompt", "qwk", "rmse",
                          "f1", "precision", "recall", "threshold"]
    # フォーマットを作る
    # 題名

    for prompt in ave_result.keys():
        for item in ave_result[prompt].keys():
            print(f"{prompt}\t{item}")
            print('\t'.join(columns_title_list))
            for train_size in [50, 100, 200, 400, 800]:
                print(train_size, end='\t')
                for column_index in columns_title_list[1:]:
                    print(ave_result[prompt][item][train_size]
                          [column_index], end='\t')
                print()
            print()
    return


def main():
    args = parse_args()
    logger.info(args)

    # print(item_list)
    # train_attention_size_list = [50, 100, 200, 400, 800]
    # attention_size_list = []

    # result = get_result(args.dir, args.include_zero_score)
    print_result(args.dir, args.include_zero_score)

    # # 設問ごと結果出力
    # if args.prompt is not None:
    #     for item in get_item_list(args.dir, args.prompt):
    #         for train_size in get_train_size_list(args.dir, args.prompt, item):
    #             for attention_size in get_attention_size_list(args.dir, args.prompt, item, train_size):
    #                 if args.prompt in result and item in result[args.prompt]:
    #                     print_per_seed(result, args.dir, args.prompt, item,
    #                                    train_size, attention_size)
    # else:
    #     # シードごとに平均した結果を取得
    #     ave_result = get_seed_ave_result(result)

    #     # # prompt itemで比較
    #     for train_size in [50, 100, 200, 400, 800]:
    #         attention_size = train_size
    #         print_per_prompt_item(
    #             ave_result, train_size, attention_size)

    #     # # 得点の訓練答案数で比較
    #     # print_per_train_size(ave_result)

    #     # # アテンションの訓練答案数で比較
    #     # print_per_attention_size(ave_result, train_size)


if __name__ == '__main__':
    main()
