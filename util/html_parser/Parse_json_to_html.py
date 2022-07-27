# encoding: utf-8

import argparse
from os import path
import itertools
import json
import util.html_parser.convert_attention_to_bin as cab
import numpy as np
import util
import part_scoring_info as psi
from os import path


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-in_file")
    parser.add_argument("-out")
    parser.add_argument("-base_file")
    parser.add_argument("-res_jsn")
    parser.add_argument("-data_jsn")
    parser.add_argument("--threshold", "-th", dest="th",
                        type=float, default=0.8)
    parser.add_argument('-dmy', action='store_true')
    parser.add_argument('-char', action='store_true')

    args = parser.parse_args()

    return args


def load_header_file(base_f):
    base_file = open(base_f)
    base_str = base_file.read()
    return base_str


def make_table_header(factors, ranges, prompt, show_evidence=False, k=1):
    base_indent = "\t\t\t"
    table = f'<th class="no">No</th>'
    table += f'<th class="ans">解答</th>'
    ded_factor = {"Miss": "誤字・脱字", "EOS": "文末"}
    for i, factor in enumerate(factors, start=2):
        if factor in ("Miss", "EOS"):
            table += "\t\t\t" + f'<th class="miss">{ded_factor[factor]}</th>\n'
        else:
            table += "\t\t\t" + \
                f'<th class="item">項目{factor}<br>({ranges[factor][prompt][1]}点)</th>\n'

    if show_evidence:
        for i, factor in enumerate(factors, start=2):
            if factor not in ("Miss", "EOS"):
                for l in range(1, k+1):
                    table += "\t\t\t" + \
                        f'<th class="ev">根拠事例{factor} {l}</th>\n'

    return table


def make_colored_response(instance, factors):
    response = instance["mecab"].split()
    response_len = len(response)
    color = ["" for i in range(len(response))]
    colored_response = ""
    for factor in factors:
        if factor in ("Miss", "EOS"):
            continue
        just = [int(j) for j in instance[factor]]
        for i, j in enumerate(just):
            if j == 1 and i < response_len:
                color[i] += factor
        if just[-1] == 1:
            color[-1] += factor

    for r, c in zip(response, color):
        if c != "":
            colored_response += f'<span class="{c}">{r}</span>'
        else:
            colored_response += r

    return colored_response


def make_colored_response_for_evidence(instance, factor):
    response = instance["mecab"].split()
    response_len = len(response)
    color = ["" for i in range(len(response))]
    colored_response = ""
    just = [int(j) for j in instance[factor].split()]
    for i, j in enumerate(just):
        if j == 1 and i < response_len:
            color[i] += factor
    if just[-1] == 1:
        color[-1] += factor

    for r, c in zip(response, color):
        if c != "":
            colored_response += f'<span class="{c}">{r}</span>'
        else:
            colored_response += r

    return colored_response


def concat_jsn(res, data, factors, char=False, dmy=False):
    concat_res = []
    for r, d in zip(res, data):
        instance = r
        instance["mecab"] = d["mecab"]
        if char:
            instance["mecab"] = " ".join(list(d["mecab"].replace(" ", "")))
        if dmy:
            instance["mecab"] += " <dmy>"
        instance["Gold"] = d["score"]
        for factor in factors:
            if factor in ("EOS", "Miss"):
                instance[factor + "_score"] = -instance[factor + "_score"]
            instance[factor + "_Gold"] = d[factor + "_Score"]

        concat_res.append(instance)
    return concat_res


def make_table_body(instances, factors, train_instances=None):
    base_indent = "\t\t\t"
    table = ""
    for k, instance in enumerate(instances, start=1):
        table += '<tr>\n'
        response = make_colored_response(instance, factors)
        table += f'\t<td>{k}</td>\n'

        table += f'\t<td class="ans">{response}</td>\n'
        for i, factor in enumerate(factors, start=2):
            score = instance[factor + "_score"]
            gold = instance[factor + "_Gold"]
            table += f'\t<td>{score} ({gold})</td>\n'
        for i, factor in enumerate(factors, start=2):
            if train_instances != None and factor not in ("Miss", "EOS"):
                assert train_instances != None, "train_instances are empty."
                evidence = train_instances[instance[factor + "_evidence"]]
                colored_evidence = make_colored_response_for_evidence(
                    evidence, factor)
                table += f'\t<td class="ev">{colored_evidence}</td>\n'

    table += "</tr>"

    return table


def make_table_body_knn(instances, factors, train_instances=None):
    base_indent = "\t\t\t"
    table = ""
    for k, instance in enumerate(instances, start=1):
        table += '<tr>\n'
        response = make_colored_response(instance, factors)
        table += f'\t<td class="no">{k}</td>\n'

        table += f'\t<td class="ans">{response}</td>\n'
        for i, factor in enumerate(factors, start=2):
            score = instance[factor + "_score"]
            gold = instance[factor + "_Gold"]
            table += f'\t<td>{score} ({gold})</td>\n'
        for i, factor in enumerate(factors, start=2):
            if train_instances != None and factor not in ("Miss", "EOS"):
                assert train_instances != None, "train_instances are empty."
                evidence_ids = instance[factor + "_evidence"]

                for id in evidence_ids:
                    evidence = train_instances[id]
                    colored_evidence = make_colored_response_for_evidence(
                        evidence, factor)
                    table += f'\t<td class="ev">{colored_evidence}</td>\n'

    table += "</tr>"

    return table


def input_result_from_json(jsn, thres):
    jsn = json.load(open(jsn, encoding='utf-8'))
    main_factors, ded_factors = get_factor(jsn)
    for instance in jsn:
        for factor in main_factors:
            just = np.array(list(map(
                float, instance[factor+"_attention"].replace("[", "").replace("]", "").split())))
            instance[factor] = cab.to_bin(just, threshold=thres)

    return jsn, (main_factors, ded_factors)


def convert_json_to_html(data_jsn, res_jsn, save_path, threshold=0.7, char=False, dmy=False):

    # filein = open(args.in_file, "r")
    base_str = load_header_file(
        path.join(path.dirname(path.abspath(__file__)), "base.html"))
    # data = [{"response":"西洋 人 は 、 自分 の 考え に 他人 を 同意 さ せる 必要 が ある が 、 西洋 文化 の 基底 に は 「 対決 」 という スタンス が あり 、 さまざま な 形 で 現われる という こと 。","Score":16,"A_score":1,"A_just":"1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0","B_score":3,"B_just":"0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0","C_score":5,"C_just":"0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0","D_score":7,"D_just":"0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"}]
    res_data, (main_factors, ded_factors) = input_result_from_json(
        res_jsn, thres=threshold)
    main_factors.extend(ded_factors)
    exp_data = json.load(open(data_jsn, encoding='utf-8'))
    prompt = psi.prompt_check(data_jsn)[0]
    ranges = psi.get_ps_ranges()
    data = concat_jsn(res_data, exp_data, main_factors, char=char, dmy=dmy)
    table = make_table_header(main_factors, ranges, prompt)
    base_str = base_str.replace("<----input_table_header_here---->", table)
    table = make_table_body(data, main_factors)
    base_str = base_str.replace("<----input_table_body_here---->", table)
    base_str = base_str.replace("<----input_prompt_name---->", prompt)
    fileout = open(save_path, "w")

    print(base_str, file=fileout)


def convert_json_to_html_with_evidence(data_jsn, res_jsn, save_path, train_data_path, threshold=0.7, char=False, dmy=False, k=1):
    # filein = open(args.in_file, "r")
    base_file_name = "base.html" if k == 1 else "base_knn.html"
    base_str = load_header_file(
        path.join(path.dirname(path.abspath(__file__)), base_file_name))
    # data = [{"response":"西洋 人 は 、 自分 の 考え に 他人 を 同意 さ せる 必要 が ある が 、 西洋 文化 の 基底 に は 「 対決 」 という スタンス が あり 、 さまざま な 形 で 現われる という こと 。","Score":16,"A_score":1,"A_just":"1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0","B_score":3,"B_just":"0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0","C_score":5,"C_just":"0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0","D_score":7,"D_just":"0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"}]
    res_data, (main_factors, ded_factors) = input_result_from_json(
        res_jsn, thres=threshold)
    train_instnaces = json.load(open(train_data_path, "r"))
    main_factors.extend(ded_factors)
    exp_data = json.load(open(data_jsn, encoding='utf-8'))
    prompt = psi.prompt_check(data_jsn)[0]
    ranges = psi.get_ps_ranges()
    data = concat_jsn(res_data, exp_data, main_factors, char=char, dmy=dmy)
    table = make_table_header(main_factors, ranges,
                              prompt, show_evidence=True, k=k)
    base_str = base_str.replace("<----input_table_header_here---->", table)
    if k == 1:
        table = make_table_body(
            data, main_factors, train_instances=train_instnaces)
    else:
        table = make_table_body_knn(
            data, main_factors, train_instances=train_instnaces)
    base_str = base_str.replace("<----input_table_body_here---->", table)
    base_str = base_str.replace("<----input_prompt_name---->", prompt)
    fileout = open(save_path, "w")

    print(base_str, file=fileout)


def convert_json_to_html_with_evidences(data_jsn, res_jsn, save_path, train_data_path, threshold=0.7, char=False, dmy=False, k=1):
    # filein = open(args.in_file, "r")
    base_str = load_header_file(
        path.join(path.dirname(path.abspath(__file__)), base_file_name))
    # data = [{"response":"西洋 人 は 、 自分 の 考え に 他人 を 同意 さ せる 必要 が ある が 、 西洋 文化 の 基底 に は 「 対決 」 という スタンス が あり 、 さまざま な 形 で 現われる という こと 。","Score":16,"A_score":1,"A_just":"1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0","B_score":3,"B_just":"0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0","C_score":5,"C_just":"0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0","D_score":7,"D_just":"0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"}]
    res_data, (main_factors, ded_factors) = input_result_from_json(
        res_jsn, thres=threshold)
    train_instnaces = json.load(open(train_data_path, "r"))
    main_factors.extend(ded_factors)
    exp_data = json.load(open(data_jsn, encoding='utf-8'))
    prompt = psi.prompt_check(data_jsn)[0]
    ranges = psi.get_ps_ranges()
    data = concat_jsn(res_data, exp_data, main_factors, char=char, dmy=dmy)
    table = make_table_header(main_factors, ranges, prompt, show_evidence=True)
    base_str = base_str.replace("<----input_table_header_here---->", table)
    table = make_table_body(
        data, main_factors, train_instances=train_instnaces)
    base_str = base_str.replace("<----input_table_body_here---->", table)
    base_str = base_str.replace("<----input_prompt_name---->", prompt)
    fileout = open(save_path, "w")

    print(base_str, file=fileout)


def main():
    args = parse_args()
    convert_json_to_html(args.base_file, args.data_jsn, args.res_jsn, args.out)
    # make_color_list(["A","B","C","D","E"])


def make_color_list(factors):
    # 色のパターンを生成する関数
    factor_num = len(factors)
    for k in range(factor_num):
        for c in itertools.combinations(factors, k+1):
            template = '.' + ''.join(list(c)) + \
                "{\n\t  background: linear-gradient("
            base = 100 // len(c)
            for i in range(len(c)):
                if i != len(c) - 1:
                    template += f"var(--{c[i]}-color){base * (i+1)}%,var(--{c[i+1]}-color){base * (i+1)}%,"
                else:
                    template += f"var(--{c[i]}-color){base * (i+1)}%"
            template += ")}"


def get_factor(jsn):
    Main = ["A", "A1", "A2", "A3", "B", "B1", "B2", "C", "D", "E", ]
    Deduction = ["Miss", "EOS"]
    main_factors = []
    ded_factors = []

    for factor in Main:
        try:
            jsn[0][factor+"_score"]
            main_factors.append(factor)
        except:
            continue
    for factor in Deduction:
        try:
            jsn[0][factor+"_score"]
            ded_factors.append(factor)
        except:
            continue

    return main_factors, ded_factors


if __name__ == '__main__':
    main()
