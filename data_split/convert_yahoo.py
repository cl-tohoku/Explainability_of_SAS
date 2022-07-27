'''
Convert movie review dataset in ERASER（https://www.eraserbenchmark.com/） to SAS json data
'''

import argparse
import logzero
from logzero import logger
import logging
from os import path, makedirs
from typing import List, Tuple
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
import random
import numpy as np
from preprocess_english_sentence import preprocess
import json
from itertools import islice
random.seed(0)


MAX_SCORE = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input movie review file path')
    parser.add_argument(
        '-o', '--output', type=path.abspath, help='output dir path')
    parser.add_argument("-bert", action="store_true",
                        default=False, help="add bert tokenize data")
    parser.add_argument(
        '-mtc', '--max_token_cnt', type=int, default=1014, help='max sentence length')
    parser.add_argument(
        '-mdcpc', '--max_data_cnt_per_cat', type=int, default=68000, help='max sentence length')
    parser.add_argument(
        "--use_category", default=["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]
    )
    parser.add_argument('--log',
                        default=None, help='Path to log file')

    args = parser.parse_args()
    return args


def load_xml_file(file_path: str):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root


def get_cont2label(root):
    ret = set()
    for data in root.iter('document'):
        # if 10 < len(ret):
        #     raise ValueError
        cat = data.find("maincat")
        if cat is not None:
            ret.add(cat.text)
        else:
            logger.warn("non category")
    return {cont: label for label, cont in enumerate(ret)}


def main():
    args = parse_args()
    if args.bert:
        from transformers import BertConfig, BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # make folder
    makedirs(args.output, exist_ok=True)

    if args.log is None:
        args.log = f"{args.output}/log.txt"
    logzero.logfile(args.log, mode='w')  # 追加: logfileの作成
    logger.info(args)

    token_cnt = []
    output_list = []

    data_cnt_per_cat = defaultdict(int)

    root = load_xml_file(args.input)
    # cont2label = get_cont2label(root)
    id = 0
    for xml_data in root.iter('document'):
        cat = xml_data.find("maincat")
        best_answer = xml_data.find("bestanswer")
        subject = xml_data.find("subject")
        content = xml_data.find("content")
        if cat is None or best_answer is None or subject is None or content is None:
            if cat is not None:
                cat = cat.text
            if best_answer is not None:
                best_answer = best_answer.text
            if subject is not None:
                subject = subject.text
            if content is not None:
                content = content.text
            logger.info(f"{cat} {best_answer} {subject} {content} has skipped")
            continue
        else:
            cat = cat.text
            best_answer = best_answer.text
            subject = subject.text
            content = content.text

        if cat not in args.use_category:
            logger.info(f"{cat} has skipped")
            continue
        else:
            if args.max_data_cnt_per_cat <= data_cnt_per_cat[cat]:
                continue
            else:
                data_cnt_per_cat[cat] += 1

        label = args.use_category.index(cat)
        best_answer = best_answer.replace("<br />", '').replace('\n', ' ')
        subject = subject.replace("<br />", '').replace('\n', ' ')
        content = content.replace("<br />", '').replace('\n', ' ')
        split_best_answer = preprocess(best_answer, is_lower=True)
        split_subject = preprocess(subject, is_lower=True)
        split_content = preprocess(content, is_lower=True)

        split_text = split_subject + split_content + split_best_answer

        # 単語数を制限
        split_text = split_text[:args.max_token_cnt]
        text = ' '.join(split_text)

        token_cnt.append(len(split_text))

        gold_justification = [0 for _ in range(len(split_text))]
        data = OrderedDict()
        data["mecab"] = text
        if args.bert:
            bert_tokenize = []
            batch_len = 256
            for batch_index in range(0, len(split_text), batch_len):
                bert_tokenize.extend(tokenizer.convert_ids_to_tokens(tokenizer(' '.join(
                    split_text[batch_index:batch_index + batch_len]), return_tensors='pt', padding=True, truncation=True)["input_ids"][0])[1:-1])
            data["bert"] = ' '.join(bert_tokenize)
            data[f"bert_A"] = ' '.join(
                map(str, [0 for _ in range(len(data["bert"].split(' ')))]))
            assert len(data["bert"].split(" ")) == len(
                data["bert_A"]), f"{len(data['bert'].split(' '))}, {len(data['bert_A'])}"

        data["A"] = ' '.join(["0" for _ in range(len(split_text))])
        data["A_Score"] = label
        data["score"] = label
        assert data["A_Score"] < MAX_SCORE, f"{data['A_Score']}"
        assert len(data["mecab"].split()) == len(data["A"].split(
        )), f'\n{len(data["mecab"].split())}: {data["mecab"]}\n{len(data["A"].split())}: {data["A"]}\n{split_text}'
        # data["id"] = int(xml_data.find("best_id").text[1:])
        data["id"] = id
        id += 1

        output_list.append(data)

    random.shuffle(output_list)
    json.dump(output_list, open(
        f"{args.output}/yahoo.json", 'w'), indent=2, ensure_ascii=False)
    logger.info(
        f"max token:{max(token_cnt)} min_token:{min(token_cnt)} ave:{np.mean(token_cnt)} sent cnt:{len(output_list)}")
    logger.info(data_cnt_per_cat)


if __name__ == '__main__':
    main()
