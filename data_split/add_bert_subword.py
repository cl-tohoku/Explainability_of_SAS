'''
Convert movie review dataset in ERASER（https://www.eraserbenchmark.com/） to SAS json data
'''

import argparse
import logzero
from logzero import logger
import logging
from os import path, makedirs
from typing import List
import json
from collections import OrderedDict
import random
import numpy as np
import copy
from tqdm import tqdm
random.seed(0)

import sys
sys.path.append("..")
from getinfo import ranges


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input json path')
    parser.add_argument(
        '-o', '--output', type=path.abspath, help='output file path')
    parser.add_argument('-c', "--config", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking",
                        choices=["bert-base-uncased", "cl-tohoku/bert-base-japanese-whole-word-masking"])
    args = parser.parse_args()
    return args


def load_jsonl_file(file_path: str):
    with open(file_path, 'r') as f:
        for line in f:
            jd = json.loads(line)
            yield jd


def main():
    args = parse_args()
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.config)

    logger.info(args)

    # make folder
    makedirs(path.dirname(args.output), exist_ok=True)

    output_list = []
    prompt = path.split(args.input)[-1].split('.')[0]

    ZEN = "".join(chr(0xff01 + i) for i in range(94))
    HAN = "".join(chr(0x21 + i) for i in range(94))
    ZEN2HAN = str.maketrans(ZEN, HAN)

    for json_data in tqdm(json.load(open(args.input))):
        text = json_data["mecab"].replace(
            ' ', '').replace('　', '●').strip("\r\n")
        # bert tokenizerのために全角を半角に変換
        text = text.translate(ZEN2HAN)
        split_text = json_data["mecab"].replace('　', '●').split(' ')

        item_list = []
        for item, v in ranges.items():
            if item == "all":
                continue
            if prompt in v:
                item_list.append(item)

        assert item_list
        for item in item_list:
            tmp = tokenizer.tokenize(text)
            # UNKを連続させないようにする
            bert_tokenize = ["dmy"]
            for token in tmp:
                if token == "[UNK]" and bert_tokenize[-1] == "[UNK]":
                    continue
                bert_tokenize.append(token)
            bert_tokenize = bert_tokenize[1:]
            json_data["bert"] = ' '.join(bert_tokenize)
            # json_data[f"bert_{item}"] =
            bert_justi = [[] for _ in range(len(json_data["bert"].split(' ')))]

            assert len(json_data["bert"].split(" ")) == len(
                bert_justi), f"{len(json_data['bert'].split(' '))}, {len(bert_justi)}"

            # gold_justification = list(map(int, json_data[item].split(' ')))
            # current_index = 0
            # current_justi = None
            # join_token = ''

            # print(split_text, bert_tokenize,
            #       gold_justification, bert_justi, sep='\n')

            # mecabのtokenizeに合わせて文字ごとのjustifiの01を取得（A_Char）
            # bertのtokenizeにjustiを分配
            # 0と1が混合してる部分は1に統一
            char_justifi = list(map(int, json_data[f"C_{item}"].split(' ')))
            char_text = ''.join(text)
            bert_index = 0
            bert_subword = bert_tokenize[bert_index].strip('#')
            for char_index in range(len(char_justifi)):
                if bert_subword == "[UNK]":
                    # 次の文字が次のbert_subwordの先頭文字ならばbert_indexを進める
                    bert_justi[bert_index].append(char_justifi[char_index])
                    if bert_index < len(bert_tokenize) - 1 and char_index < len(char_text) - 1 and bert_tokenize[bert_index + 1].strip('#').startswith(char_text[char_index + 1]):
                        bert_index += 1
                    bert_subword = bert_tokenize[bert_index].strip('#')
                elif bert_subword.startswith(char_text[char_index]):
                    bert_justi[bert_index].append(char_justifi[char_index])
                    bert_subword = bert_subword[1:]
                else:
                    raise ValueError(
                        f"bert_subword: {bert_subword}\t {char_text[char_index]}")

                if not bert_subword and bert_index < len(bert_tokenize) - 1:
                    bert_index += 1
                    bert_subword = bert_tokenize[bert_index].strip('#')

            bert_justi = list(map(lambda x: 1 if 1 in x else 0, bert_justi))

            json_data[f"bert_{item}"] = ' '.join(map(str, bert_justi))

        output_list.append(copy.deepcopy(json_data))

    json.dump(output_list, open(
        args.output, 'w'), indent=2, ensure_ascii=False)


# def _remove(join_token, _bert_justi, bert_tokenize, current_justi, current_index):
#     bert_justi = copy.deepcopy(_bert_justi)
#     while len(join_token) != 0:
#         bert_justi[current_index] = current_justi
#         remove_token = bert_tokenize[current_index].strip("#")
#         if remove_token == "[UNK]":
#             # bertは1文字をUNKに置き換えると仮定
#             remove_token = join_token[0]
#         assert join_token.startswith(
#             remove_token), f"join_token: {join_token}\t remove_token: {remove_token}"
#         join_token = join_token[len(remove_token):]
#         current_index += 1
#     return bert_justi, current_index


if __name__ == '__main__':
    main()
