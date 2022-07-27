import click
import json
import os
import sys
import numpy as np
sys.path.append('..')
from sas.quadratic_weighted_kappa import quadratic_weighted_kappa as QWK
from getinfo import ranges, limit_leng


def _calc_justi_agreement(rater1, rater2, epsilon=1e-5):
    assert len(rater1) == len(rater2), f'''{len(rater1)}, {len(rater2)}'''
    if sum(rater1) == 0 or sum(rater2) == 0:
        # どちらかが0点答案のときは計算しない
        return np.nan
    TP = sum([r1 * r2 for r1, r2 in zip(rater1, rater2)])
    FP = sum([r1 == 1 and r2 == 0 for r1, r2 in zip(rater1, rater2)])
    FN = sum([r1 == 0 and r2 == 1 for r1, r2 in zip(rater1, rater2)])
    recall = TP / (TP + FP)
    precision = TP / (TP + FN)
    f1 = 2 * (recall * precision) / (recall + precision + epsilon)
    return f1


def get_char_justi(sent, justi):
    split_sent = sent.split(' ')
    split_justi = list(map(int, justi.split(' ')))
    assert len(split_sent) == len(split_justi)
    char_justi = []
    for token, justi in zip(split_sent, split_justi):
        if justi == 1:
            char_justi += [1] * len(token)
        else:
            char_justi += [0] * len(token)

    return char_justi


def calc_justi_agreement_sequency(rater1_list, rater2_list):
    assert len(rater1_list) == len(rater2_list)
    return np.nanmean([_calc_justi_agreement(rater1, rater2) for rater1, rater2 in zip(rater1_list, rater2_list)])


@click.group()
def cli():
    pass


# @cli.command()
# @click.option('--rater1_json_path', type=click.Path(exists=True), help="1人めのアノテータのjson file")
# @click.option('--rater2_json_path', type=click.Path(exists=True), help="2人めのアノテータのjson file")
# @click.option('--item', type=str, help="採点項目")
# @click.option('--prompt', type=str, help="設問の名前")
def calc_QWK_agreement(rater1_json_path, rater2_json_path, item, prompt):
    score_key = f"{item}_Score"
    rater1data = [data[score_key]
                  for data in json.load(open(rater1_json_path))]
    rater2data = [data[score_key]
                  for data in json.load(open(rater2_json_path))]
    min_score, max_score = ranges[item][prompt]

    # from collections import Counter
    # print(Counter(rater1data))
    # print(Counter(rater2data))
    qwk = QWK(rater1data, rater2data,
              min_rating=min_score, max_rating=max_score)

    # print(qwk)
    return qwk


def get_justification(rater_json_path, item, add_dmy):
    key = f"{item}"
    json_data = json.load(open(rater_json_path))
    # assert key in json_data[0], f'''{key} not include in {json_data[0].keys()}'''
    raterdata = [list(map(int, get_char_justi(data["mecab"], data[key])))  # + [int('1' not in data[key])]
                 for data in json_data]
    if add_dmy:
        raterdata = [data + [int(1 not in data)] for data in raterdata]
    return raterdata

# @cli.command()
# @click.option('--rater1_json_path', type=click.Path(exists=True), help="1人めのアノテータのjson file")
# @click.option('--rater2_json_path', type=click.Path(exists=True), help="2人めのアノテータのjson file")
# @click.option('--item', type=str, help="採点項目")


def calc_justi_agreement(rater1_json_path, rater2_json_path, item, add_dmy):
    rater1data = get_justification(rater1_json_path, item, add_dmy)
    rater2data = get_justification(rater2_json_path, item, add_dmy)

    justi_agreement = calc_justi_agreement_sequency(rater1data, rater2data)
    # print(justi_agreement)
    return justi_agreement


def get_ave_sent_len(rater_json_path, item, add_dmy):
    # key = f"{item}"
    # ret = np.mean([len(data[key].split(' '))
    #                for data in json.load(open(rater_json_path))])
    ret = np.mean([len(data)
                   for data in get_justification(rater_json_path, item, add_dmy)])
    return ret


def get_ave_justi_sent_len(rater_json_path: str, item: str, add_dmy: bool) -> int:
    key = f"{item}"
    justi_list = [list(map(int, get_char_justi(data["mecab"], data[key])))
                  for data in json.load(open(rater_json_path))]
    if add_dmy:
        justi_list = [data + [int(1 not in data)] for data in justi_list]

    ret = np.nanmean(
        [sum(data) if 1 in data else np.nan for data in justi_list])
    return ret


@cli.command()
@click.option('--rater1_json_path', type=click.Path(exists=True), help="1人めのアノテータのjson file")
@click.option('--rater2_json_path', type=click.Path(exists=True), help="2人めのアノテータのjson file")
@click.option('--data_json_path', type=click.Path(exists=True), help="元データのjson file")
@click.option('--item', type=str, help="採点項目")
@click.option('--prompt', type=str, help="設問の名前")
@click.option('--dmy', is_flag=True, default=False, help="dmy　tokenを入れるか")
def calc_dataset_info(rater1_json_path, rater2_json_path, data_json_path, item, prompt, dmy):
    qwk = calc_QWK_agreement(rater1_json_path, rater2_json_path, item, prompt)
    avg_sent_len = get_ave_sent_len(data_json_path, item, False)
    avg_justi_sent_len = get_ave_justi_sent_len(
        data_json_path, item, False)

    justi_agreement = calc_justi_agreement(
        rater1_json_path, rater2_json_path, item, dmy)

    # ll = limit_leng[prompt]

    print(prompt, item, avg_sent_len,
          avg_justi_sent_len, qwk, justi_agreement)


if __name__ == "__main__":
    cli()
