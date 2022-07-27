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
random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input movie review dir path')
    parser.add_argument(
        '-o', '--output', type=path.abspath, help='output dir path')
    parser.add_argument("-bert", action="store_true",
                        default=False, help="add bert tokenize data")
#    parser.add_argument('--log',
#                        default="log.txt", help='Path to log file')
    args = parser.parse_args()
    return args


def load_jsonl_file(file_path: str):
    with open(file_path, 'r') as f:
        for line in f:
            jd = json.loads(line)
            yield jd


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
    for mode in ["train", 'dev', 'test']:
        output_list = []
        for j_d in load_jsonl_file(f"{args.input}/{mode}.jsonl"):
            with open(f"{args.input}/docs/{j_d['annotation_id']}", 'r') as ft:
                text = ' '.join(
                    map(lambda x: x.strip("\r\n"), ft.readlines()))
                split_text = text.split(" ")
                token_cnt.append(len(split_text))
            # load jsonl file which include justification cue, and posinega and reference text
            gold_justification = [0 for _ in range(len(split_text))]
            data = OrderedDict()
            data["mecab"] = text
            if args.bert:
                bert_tokenize = []
                batch_len = 256
                for batch_index in range(0,len(split_text),batch_len):
                    bert_tokenize.extend(tokenizer.convert_ids_to_tokens(tokenizer(' '.join(split_text[batch_index:batch_index+batch_len]), return_tensors='pt',padding=True, truncation=True)["input_ids"][0])[1:-1])
                data["bert"] = ' '.join(bert_tokenize)
                data[f"bert_A"] = [0 for _ in range(len(data["bert"].split(' ')))]
                assert len(data["bert"].split(" ")) == len(data["bert_A"]), f"{len(data['bert'].split(' '))}, {len(data['bert_A'])}"


            data["A_Score"] = 1 if j_d["classification"] == "POS" else 0
            data["score"] = 1 if j_d["classification"] == "POS" else 0
            data["id"] = int(j_d["annotation_id"][5:8])

            if j_d["evidences"]:
                for evidence in j_d["evidences"]:
                    evidence = evidence[0]
                    start = evidence["start_token"]
                    end = evidence["end_token"]
                    assert ' '.join(
                        split_text[start:end]) == evidence["text"], f"text no matching\n{split_text[start:end]}\n{ evidence['text']}"
                    for i in range(start, end):
                        gold_justification[i] = 1

                if args.bert:
                    start = 0
                    end = 0
                    for i in range(len(gold_justification)):
                        # join_token = []
                        join_token = [(bert_tokenize[end])]
                        while True:
                            end += 1
                            if len(bert_tokenize) <= end or split_text[i] == ''.join(join_token) or "[UNK]" in join_token:
                                break
                            if bert_tokenize[end].startswith("##"):
                                join_token.append(bert_tokenize[end][2:])
                            elif len(''.join(join_token)) < len(split_text[i]):
                                # mecabとbertのトークナイズでそういすること路がある
                                join_token.append(bert_tokenize[end])
                            else: 
                                logger.error(f"{split_text[i]}, {''.join(join_token)}")
                                raise ValueError
                        # assert split_text[i] == ''.join(join_token), f"{split_text[i]}, {''.join(join_token)}"
                        for index in range(start, end):
                            data["bert_A"][index] = gold_justification[i]
                        start = end

                    data[f"bert_A"] = ' '.join(map(str,data["bert_A"]))

            else:
                logger.warning("evidence list is empty")
            data["A"] = ' '.join(map(str, gold_justification))
            output_list.append(data)


        random.shuffle(output_list)
        json.dump(output_list, open(
            f"{args.output}/Movie_Reviews_{mode}.json", 'w'), indent=2, ensure_ascii=False)
    logger.info(
        f"max token:{max(token_cnt)} min_token:{min(token_cnt)} ave:{np.mean(token_cnt)}")


if __name__ == '__main__':
    main()
