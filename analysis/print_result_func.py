from os import path
from typing import List, Dict
from glob import glob


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
