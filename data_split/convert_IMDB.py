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
from preprocess_english_sentence import preprocess
random.seed(0)
MAX_SCORE=10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input movie review file path')
    parser.add_argument(
        '-o', '--output', type=path.abspath, help='output dir path')
    parser.add_argument("-bert", action="store_true",
                        default=False, help="add bert tokenize data")
    parser.add_argument(
        '-max', '--max_length', type=int, default=100000000000, help='max sentence length')
#    parser.add_argument('--log',
#                        default="log.txt", help='Path to log file')
    args = parser.parse_args()
    return args

def load_json_file(file_path: str):
    with open(file_path, 'r') as f:
        jd = json.load(f)
    for jsonl in jd:
        yield jsonl


def main():
    args = parse_args()
    if args.bert:
        from transformers import BertConfig, BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)

    # make folder
    makedirs(args.output, exist_ok=True)

    token_cnt = []
    output_list = []
    
    for id,j_d in enumerate(load_json_file(args.input)):
        split_text = preprocess(j_d["review"],remove_stopwords=False, is_steming=False, is_lower=True) 
        # split_text = list(map(lambda x:x.lower(),j_d["review"].split()))

        text = ' '.join(split_text)
        if args.max_length < len(text):
            continue
        token_cnt.append(len(split_text))

        gold_justification = [0 for _ in range(len(split_text))]
        data = OrderedDict()
        data["mecab"] = text
        if args.bert:
            bert_tokenize = []
            batch_len = 256
            for batch_index in range(0,len(split_text),batch_len):
                bert_tokenize.extend(tokenizer.convert_ids_to_tokens(tokenizer(' '.join(split_text[batch_index:batch_index+batch_len]), return_tensors='pt',padding=True, truncation=True)["input_ids"][0])[1:-1])
            data["bert"] = ' '.join(bert_tokenize)
            data[f"bert_A"] = ' '.join(map(str,[0 for _ in range(len(data["bert"].split(' ')))]))
            assert len(data["bert"].split(" ")) == len(data["bert_A"]), f"{len(data['bert'].split(' '))}, {len(data['bert_A'])}"


        data["A"] = ' '.join(["0" for _ in range(len(split_text))])
        data["A_Score"] = j_d["rating"] - 1
        data["score"] = j_d["rating"] - 1 
        assert data["A_Score"] < MAX_SCORE, f"{data['A_Score']}"
        assert len(data["mecab"].split()) == len(data["A"].split()), f'\n{len(data["mecab"].split())}: {data["mecab"]}\n{len(data["A"].split())}: {data["A"]}\n{split_text}'
        data["id"] = id

        output_list.append(data)

    random.shuffle(output_list)
    json.dump(output_list, open(
        f"{args.output}/IMDB.json", 'w'), indent=2, ensure_ascii=False)
    logger.info(
        f"max token:{max(token_cnt)} min_token:{min(token_cnt)} ave:{np.mean(token_cnt)} sent cnt:{len(output_list)}")


if __name__ == '__main__':
    main()
