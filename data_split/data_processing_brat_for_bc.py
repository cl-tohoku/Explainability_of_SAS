"""
usage:
    python data_processing_brat_for_bc.py -d ~/University/Lab/SAS/CRLEA/20210905新規データ5問分/1_12/1_12.txt -b ~/University/Lab/SAS/CRLEA/20210905新規データ5問分/1_12/1_12.ann --prefix ~/Desktop/1_12 --part-names A B SPL EOS     
"""

import argparse
import codecs
import json
import collections as cl
import os

two_factor_list = ["Y14_2_2-3_2_C"]
three_factor_list = ["Y15_1-3_1_2"]
four_factor_list = ["Y14_1_1_1_3", "Y14_1_2_1_3", "Y14_1_2_2_4",
                    "Y14_2_1_2_3", "Y15_2_3_1_5", "Y15_2_3_2_2", "Y15_2_3_2_4"]
eos_factor_list = ["Y14_1_1_1_3", "Y14_1_1_2_5", "Y14_1_2_1_3",
                   "Y14_1_2_2_4", "Y14_2_1_1_5", "Y15_2_3_1_4", "Y15_2_3_2_4", "Y15_1-3_1_2"]
five_factor_list = ["Y15_2_3_1_5", "Y15_2_3_2_2", "Y15_2_3_2_4", ]


def main():
    args = parse_args()
    brat_data = args.brat_data
    data = args.data_path
    # json_data = args.json_data
    # five_factor, four_factor, three_factor, two_factor, eos_factor = prompt_check(
    #     brat_data)
    # part_names = get_item_list(
    #     five_factor, four_factor, three_factor, two_factor, eos_factor)
    part_names = args.part_names
    if "EOS" in part_names:
        eos_factor = True

    # 答案を読み込む
    input_list = []
    with codecs.open(data, 'r', encoding='utf8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.startswith("END"):
                input_list.append(line)

    # with codecs.open(json_data, 'r', encoding='utf8') as f:
    #     json_texts = json.load(f)

    # bratのアノテーション情報を読み込む
    brat_list = []
    with codecs.open(brat_data, 'r', encoding='utf8') as f:
        for line in f:
            line = line.rstrip('\n')
            brat_list.append(line)

    # {"key	item start end  答案":start}
    brat_dict_cue = {}
    # {"key":point}
    brat_dict_points = {}
    for brat_infos in brat_list:
        brat_info = brat_infos.split("\t")
        if brat_infos.startswith("T"):
            part_info, start, end = brat_info[1].split(" ")
            val = int(start)
            brat_dict_cue[brat_infos] = val
        elif brat_infos.startswith("A"):
            info, key, point = brat_info[1].split(" ")
            brat_dict_points[key] = int(point)
    # startが早い順番にソート
    brat_dict_cue = sorted(brat_dict_cue.items(), key=lambda x: x[1])

    strt_idx = 0
    ann_num = 0
    json_list = []

    for num, sent in enumerate(input_list):
        data = cl.OrderedDict()
        end_idx = strt_idx + len(sent)
        words = sent.split(" ")
        chars = list(sent.replace(" ", ''))
        ann_flag = False
        scores = {}
        lists = {}
        parsed = {}
        # id = json_texts[num]["id"]
        id = num + 1

        for p_name in part_names:
            scores[p_name] = 0
            lists[p_name] = ["0" for j in range(len(words))]
            parsed[p_name] = ""

        while not ann_flag and ann_num < len(brat_dict_cue):
            if brat_dict_cue[ann_num][1] >= strt_idx and brat_dict_cue[ann_num][1] <= end_idx:
                char_word_index = make_char_word_index(sent)
                brat_info = brat_dict_cue[ann_num][0].split("\t")
                key = brat_info[0]
                part_info, start, end = brat_info[1].split(" ")
                start = int(start) - strt_idx
                end = int(end) - strt_idx
                length = end - start
                if key in brat_dict_points:
                    point = brat_dict_points[key]
                    scores[part_info] = point
                for j in range(length):
                    num = j + start
                    # print(char_word_index[num], end=' ')
                    lists[part_info][char_word_index[num]] = "1"
                ann_num += 1
            else:
                ann_flag = True
        strt_idx = end_idx + 1

        # 1より大きいものは全て1にする
        if scores["SPL"] >= 1:
            scores["SPL"] = 1
        score = 0

        for p_name in part_names:
            # 全体点の計算
            if p_name in ["SPL", "EOS"]:
                score -= scores[p_name]
            else:
                score += scores[p_name]

        # scoreは負にしない
        if score < 0:
            score = 0

        # /r
        words.pop()
        # SPL
        words.pop()
        for p_name in part_names:
            lists[p_name].pop()
            lists[p_name].pop()

        if eos_factor:
            words.pop()
            for p_name in part_names:
                lists[p_name].pop()
        if args.score:
            words = words[1:]
            for p_name in part_names:
                lists[p_name].pop(0)
        for p_name in part_names:
            parsed[p_name] = " ".join(lists[p_name])

        sent = " ".join(words)
        data["mecab"] = sent
        data["score"] = score
        # 追加
        data['Char'] = ' '.join(list(''.join(words)))

        for p_name in part_names:
            if p_name == "SPL":
                data['Miss_Score'] = - scores[p_name]
            elif p_name == "EOS":
                data['EOS_Score'] = - scores[p_name]
            else:
                data[p_name] = parsed[p_name]
                key = p_name + "_Score"
                data[key] = scores[p_name]

                temp = []
                for word, attn in zip(data["mecab"].split(), data[p_name].split()):
                    for char in word:
                        temp.append(attn)
                data["C_" + p_name] = ' '.join(temp)

        data['id'] = id
        json_list.append(data)
    file_name = args.prefix + ".json"
    os.makedirs(os.path.split(args.prefix)[0], exist_ok=True)
    fw = open(file_name, 'w')
    json.dump(json_list, fw, indent=2, ensure_ascii=False)


def make_char_word_index(parsed):
    word_id = 0
    char_word_index = []
    for i in range(len(parsed)):
        char_word_index.append(word_id)
        if parsed[i] == " ":
            word_id += 1
    return char_word_index


def prompt_check(file):
    five_factor, four_factor, three_factor, two_factor, eos_factor = False, False, False, False, False
    for p in five_factor_list:
        if p in file:
            five_factor = True
            four_factor = True
            break
    for p in four_factor_list:
        if p in file:
            four_factor = True
            break
    for p in three_factor_list:
        if p in file:
            three_factor_factor = True
            break
    for p in two_factor_list:
        if p in file:
            two_factor = True
            break
    for p in eos_factor_list:
        if p in file:
            eos_factor = True
            break
    return five_factor, four_factor, three_factor, two_factor, eos_factor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path", type=str,
                        metavar='<str>', required=True, help="The path to the data (.txt)")
    parser.add_argument("-b", "--brat_data", dest="brat_data", type=str,
                        metavar='<str>', required=True, help="The path to the brat data (.ann)")
    # parser.add_argument("-j", "--json_data", dest="json_data", type=str,
    #                     metavar='<str>', required=True, help="The path to the json data (.json)")
    parser.add_argument("-p", "--prefix", dest="prefix", type=str,
                        metavar='<str>', required=True, help="The prefix of file name")
    parser.add_argument("--part-names", "-pn", dest="part_names",
                        nargs="*", type=str, help="Item names. ")
    parser.add_argument("-score", dest="score",
                        default=False, action='store_true', help="")

    args = parser.parse_args()
    return args

#
# def get_item_list(five_factor, four_factor, three_factor, two_factor, eos_factor):
#     if two_factor:
#         part_names = ["A", "SPL"]
#         return part_names
#
#     part_names = ["A", "B", "C"]
#     if five_factor and eos_factor:
#         part_names.extend(["D", "E", "SPL", "EOS"])
#     elif five_factor:
#         part_names.extend(["D", "E", "SPL"])
#     elif four_factor and eos_factor:
#         part_names.extend(["D", "SPL", "EOS"])
#     elif four_factor:
#         part_names.extend(["D", "SPL"])
#     elif eos_factor:
#         part_names.extend(["SPL", "EOS"])
#     else:
#         part_names.extend(["SPL"])
#     return part_names


if __name__ == '__main__':
    main()
