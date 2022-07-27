'''
    浅妻実験で用いられているデータを自動採点モデルで回せるようにする
'''

import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
import pandas as pd
import os
from collections import OrderedDict

from transformers import BertConfig, BertTokenizer
import re
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input file path')
    parser.add_argument(
        '-o', '--output', type=path.abspath, help='output directory')
    args = parser.parse_args()
    return args

class Creator:
    def __init__(self):
        self.id = 0
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __call__(self, text, label):
        split_text = text.split(' ')
        gold_justification = [0 for _ in range(len(split_text))]
        data = OrderedDict()
        data["mecab"] = text

        bert_tokenize = []
        batch_len = 256
        for batch_index in range(0,len(split_text),batch_len):
            bert_tokenize.extend(self.tokenizer.convert_ids_to_tokens(self.tokenizer(' '.join(split_text[batch_index:batch_index+batch_len]), return_tensors='pt',padding=True, truncation=True)["input_ids"][0])[1:-1])
        data["bert"] = ' '.join(bert_tokenize)
        data[f"bert_A"] = ' '.join(map(str,[0 for _ in range(len(data["bert"].split(' ')))]))
        assert len(data["bert"].split(" ")) == len(data["bert_A"].split(' ')), f"{len(data['bert'].split(' '))}, {len(data['bert_A'])}\n{data['bert'].split(' ')}\n {data['bert_A']}"


        data["A"] = ' '.join(["0" for _ in range(len(split_text))])
        data["A_Score"] = label
        data["score"] = label
        # assert data["A_Score"] < MAX_SCORE, f"{data['A_Score']}"
        assert len(data["mecab"].split()) == len(data["A"].split()), f'\n{len(data["mecab"].split())}: {data["mecab"]}\n{len(data["A"].split())}: {data["A"]}\n{split_text}'
        data["id"] = self.get_id()

        return data

    def get_id(self):
        ret = self.id
        self.id += 1
        return ret

def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)
    creator = Creator()

    df = pd.read_csv(args.input)
    name = os.path.split(args.input)[1].split("_dataset.csv")[0]

        
    output = {"train":[], "dev":[], "test":[]}

    for i in range(len(df)):
        label = int(df.iloc[i]["label"])
        text = df.iloc[i]["text"]
        text = text.replace("\x85"," . ")
        text = text.replace("\x96"," . ")
        text = text.replace("\x97"," ")
        text = re.sub(r'\s+', lambda x:' ', text)
        text = text.strip()
        mode = df.iloc[i]["exp_split"]

        data = creator(text, label)

        output[mode].append(data)

    dev_file = f"{args.output}/{name}/{name}_dev.json"
    test_file = f"{args.output}/{name}/{name}_test.json"
    train_file = f"{args.output}/{name}/data_size/{name}_train.{len(output['train'])}.0.json"
    os.makedirs(f"{args.output}/{name}/data_size", exist_ok=True)

    with open(train_file,'w') as f_train, open(dev_file,'w') as f_dev, open(test_file,'w') as f_test:
        json.dump(output['train'], f_train, indent=2)
        json.dump(output['dev'], f_dev, indent=2)
        json.dump(output['test'], f_test, indent=2)


if __name__ == '__main__':
    main()
