'''
ranges: lowerbount and upperbound
four_factor_list: prompts have four factor
eos_factor_list: prompts have deduction by EOS expression
'''

import numpy as np

Y14_1_1_1_3 = "Y14_1-1_1_3"
Y14_1_1_2_5 = "Y14_1-1_2_5"
Y14_1_2_1_3 = "Y14_1-2_1_3"
Y14_1_2_2_4 = "Y14_1-2_2_4"
Y14_2_1_1_5 = "Y14_2-1_1_5"
Y14_2_1_2_3 = "Y14_2-1_2_3"
Y14_2_2_1_4 = "Y14_2-2_1_4"
Y14_2_2_2_3 = "Y14_2-2_2_3"
Y15_2_3_1_4 = "Y15_2-3_1_4"
Y15_2_3_1_5 = "Y15_2-3_1_5"
Y15_2_3_2_2 = "Y15_2-3_2_2"
Y15_2_3_2_4 = "Y15_2-3_2_4"
tmp_exp_data = "tmp_exp_data"
pretest_ocr_1_q1 = "pretest_ocr_1-q1"
pretest_ocr_1_q2 = "pretest_ocr_1-q2"
pretest_ocr_1_q3 = "pretest_ocr_1-q3"
# 2019 data
Y14_2_2_3_2_C = "Y14_2_2-3_2_C"
Y14_2_2_1_5_G = "Y14_2_2-1_5_G"
Y14_2_2_3_3_G = "Y14_2_2-3_3_G"
Y15_1_3_1_2 = "Y15_1-3_1_2"
Y15_1_3_1_5 = "Y15_1-3_1_5"
Y15_1_3_2_4 = "Y15_1-3_2_4"
Y15_1_3_2_5 = "Y15_1-3_2_5"
Y15_1_1_1_4 = "Y15_1-1_1_4"
Y15_1_1_1_6 = "Y15_1-1_1_6"
Y15_1_1_2_4 = "Y15_1-1_2_4"
Y15_1_1_2_5 = "Y15_1-1_2_5"
Y15_2_2_1_3 = "Y15_2-2_1_3"
Y15_2_2_1_5 = "Y15_2-2_1_5"
Y15_2_2_2_4 = "Y15_2-2_2_4"
Y15_2_2_2_5 = "Y15_2-2_2_5"
Y16_2_3_5_2 = "Y16_2-3_5_2"
Y17_M_1_6 = "Y17_M-1_6"

# CRLEA data
B1_2 = "B1_2"
B1_3 = "B1_3"
B1_5 = "B1_5"
B2_7 = "B2_7"
B2_8 = "B2_8"
B2_9 = "B2_9"
B2_10 = "B2_10"
B2_11 = "B2_11"
B1_12 = "B1_12"
B1_13 = "B1_13"
B1_14 = "B1_14"
B1_15 = "B1_15"
B1_16 = "B1_16"

# ERASER
Movie_Reviews = "Movie_Reviews"

# IMDB
IMDB = "IMDB"

# yahoo
yahoo = "yahoo"

# asazuma
imdb = "imdb"
sst = 'sst'
agnews = "agnews"
_20news = "20News_sports"

prompt_list = [Y14_1_1_1_3,
               Y14_1_1_2_5,
               Y14_1_2_1_3,
               Y14_1_2_2_4,
               Y14_2_1_1_5,
               Y14_2_1_2_3,
               Y14_2_2_1_4,
               Y14_2_2_2_3,
               Y15_2_3_1_4,
               Y15_2_3_1_5,
               Y15_2_3_2_2,
               Y15_2_3_2_4,
               tmp_exp_data,
               pretest_ocr_1_q1,
               pretest_ocr_1_q2,
               pretest_ocr_1_q3,
               Y14_2_2_3_2_C,
               Y14_2_2_1_5_G,
               Y14_2_2_3_3_G,
               Y15_1_3_1_2,
               Y15_1_3_1_5,
               Y15_1_3_2_4,
               Y15_1_3_2_5,
               Y15_1_1_1_4,
               Y15_1_1_1_6,
               Y15_1_1_2_4,
               Y15_1_1_2_5,
               Y15_2_2_1_3,
               Y15_2_2_1_5,
               Y15_2_2_2_4,
               Y15_2_2_2_5,
               Y16_2_3_5_2,
               Y17_M_1_6,
               B1_2,
               B1_3,
               B1_5,
               B2_7,
               B2_8,
               B2_9,
               B2_10,
               B2_11,
               B1_12,
               B1_13,
               B1_14,
               B1_15,
               B1_16,
               Movie_Reviews,
               IMDB,
               yahoo,
               imdb,
               sst,
               agnews,
               _20news,
               ]

ranges = {
    "all": {
        Y14_1_1_1_3: (0, 15),
        Y14_1_1_2_5: (0, 14),
        Y14_1_2_1_3: (0, 16),
        Y14_1_2_2_4: (0, 12),
        Y14_2_1_1_5: (0, 15),
        Y14_2_1_2_3: (0, 12),
        Y14_2_2_1_4: (0, 15),
        Y14_2_2_2_3: (0, 14),
        Y15_2_3_1_4: (0, 14),
        Y15_2_3_1_5: (0, 16),
        Y15_2_3_2_2: (0, 12),
        Y15_2_3_2_4: (0, 14),
        tmp_exp_data: (0, 1),
        pretest_ocr_1_q1: (0, 4),
        pretest_ocr_1_q2: (0, 4),
        pretest_ocr_1_q3: (0, 4),
        Y14_2_2_3_2_C: (0, 4),
        Y14_2_2_1_5_G: (0, 2),
        Y14_2_2_3_3_G: (0, 2),
        Y15_1_3_1_2: (0, 14),
        Y15_1_3_1_5: (0, 16),
        Y15_1_3_2_4: (0, 9),
        Y15_1_3_2_5: (0, 10),
        Y15_1_1_1_4: (0, 15),
        Y15_1_1_1_6: (0, 15),
        Y15_1_1_2_4: (0, 12),
        Y15_1_1_2_5: (0, 12),
        Y15_2_2_1_3: (0, 12),
        Y15_2_2_1_5: (0, 14),
        Y15_2_2_2_4: (0, 12),
        Y15_2_2_2_5: (0, 12),
        Y16_2_3_5_2: (0, 12),
        Y17_M_1_6: (0, 15),
        B1_2: (0, 6),
        B1_3: (0, 6),
        B1_5: (0, 9),
        B2_7: (0, 7),
        B2_8: (0, 8),
        B2_9: (0, 6),
        B2_11: (0,),
        B2_10: (0, 8),
        B1_12: (0, 6),
        B1_13: (0, 3),
        B1_14: (0, 6),
        B1_15: (0, 6),
        B1_16: (0, 5),
        Movie_Reviews: (0, 1),
        IMDB: (0, 9),
        yahoo: (0, 9),
        imdb: (0, 1),
        sst: (0, 1),
        agnews: (0, 1),
        _20news: (0, 1),
    },
    "A": {
        Y14_1_1_1_3: (0, 2),
        Y14_1_1_2_5: (0, 5),
        Y14_1_2_1_3: (0, 2),
        Y14_1_2_2_4: (0, 3),
        Y14_2_1_1_5: (0, 2),
        Y14_2_1_2_3: (0, 3),
        Y14_2_2_1_4: (0, 6),
        Y14_2_2_2_3: (0, 6),
        Y15_2_3_1_4: (0, 5),
        Y15_2_3_1_5: (0, 3),
        Y15_2_3_2_2: (0, 3),
        Y15_2_3_2_4: (0, 2),
        Y14_2_2_3_2_C: (0, 4),
        Y15_1_1_2_4: (0, 5),
        Y15_1_1_2_5: (0, 3),
        Y15_2_2_1_3: (0, 4),
        Y15_2_2_1_5: (0, 7),
        Y15_1_3_1_2: (0, 4),
        Y15_1_3_1_5: (0, 6),
        Y15_1_3_2_4: (0, 5),
        Y15_1_3_2_5: (0, 4),
        Y15_1_1_1_4: (0, 2),
        Y15_1_1_1_6: (0, 5),
        Y15_1_3_2_4: (0, 5),
        Y15_2_2_2_4: (0, 3),
        Y15_2_2_2_5: (0, 4),
        B1_2: (0, 2),
        B1_3: (0, 2),
        B1_5: (0, 3),
        # B2_7: (0, 3),
        # B2_8: (0, 3),
        # B2_9: (0,),
        # B2_11: (0,),
        B2_10: (0, 4),
        B1_12: (0, 3),
        B1_13: (0, 3),
        B1_15: (0, 2),
        B1_16: (0, 2),
        Movie_Reviews: (0, 1),
        IMDB: (0, 9),
        yahoo: (0, 9),
        imdb: (0, 1),
        sst: (0, 1),
        agnews: (0, 1),
        _20news: (0, 1),
    },
    "B": {
        Y14_1_1_1_3: (0, 4),
        Y14_1_1_2_5: (0, 4),
        Y14_1_2_1_3: (0, 5),
        Y14_1_2_2_4: (0, 2),
        Y14_2_1_1_5: (0, 7),
        Y14_2_1_2_3: (0, 4),
        Y14_2_2_1_4: (0, 3),
        Y14_2_2_2_3: (0, 6),
        Y15_2_3_1_4: (0, 5),
        Y15_2_3_1_5: (0, 3),
        Y15_2_3_2_2: (0, 2),
        Y15_2_3_2_4: (0, 3),
        Y15_1_1_2_4: (0, 5),
        Y15_1_1_2_5: (0, 7),
        Y15_2_2_1_3: (0, 4),
        Y15_2_2_1_5: (0, 7),
        Y15_1_3_2_4: (0, 4),
        Y15_1_3_2_5: (0, 6),
        Y15_1_3_1_2: (0, 5),
        Y15_1_3_1_5: (0, 10),
        Y15_1_1_1_4: (0, 5),
        Y15_1_1_1_6: (0, 10),
        Y15_1_3_2_4: (0, 4),
        Y15_2_2_2_4: (0, 5),
        Y15_2_2_2_5: (0, 3),
        B1_2: (0, 2),
        B1_3: (0, 2),
        B1_5: (0, 4),
        B2_10: (0, 4),
        B2_7: (0, 4),
        B2_8: (0, 3),
        B2_9: (0, 2),
        B1_12: (0, 3),
        B1_14: (0, 3),
        B1_15: (0, 2),
        B1_16: (0, 3),
        # B2_11: (0,),

    },
    "C": {
        Y14_1_1_1_3: (0, 4),
        Y14_1_1_2_5: (0, 5),
        Y14_1_2_1_3: (0, 3),
        Y14_1_2_2_4: (0, 4),
        Y14_2_1_1_5: (0, 6),
        Y14_2_1_2_3: (0, 3),
        Y14_2_2_1_4: (0, 6),
        Y14_2_2_2_3: (0, 2),
        Y15_2_3_1_4: (0, 4),
        Y15_2_3_1_5: (0, 2),
        Y15_2_3_2_2: (0, 2),
        Y15_2_3_2_4: (0, 3),
        Y15_1_1_2_4: (0, 2),
        Y15_1_1_2_5: (0, 2),
        Y15_1_3_1_2: (0, 5),
        Y15_2_2_1_3: (0, 4),
        Y15_1_1_1_4: (0, 8),
        Y15_2_2_2_4: (0, 4),
        Y15_2_2_2_5: (0, 5),
        B1_2: (0, 2),
        B1_5: (0, 2),
        # B2_8: (0, 3),
        B2_11: (0, 2),
        B1_15: (0, 2),

    },
    "D": {
        Y14_1_1_1_3: (0, 5),
        Y14_1_2_1_3: (0, 6),
        Y14_1_2_2_4: (0, 3),
        Y14_2_1_2_3: (0, 2),
        Y15_2_3_1_5: (0, 4),
        Y15_2_3_2_2: (0, 3),
        Y15_2_3_2_4: (0, 4)
    },
    "E": {
        Y15_2_3_1_5: (0, 4),
        Y15_2_3_2_2: (0, 2),
        Y15_2_3_2_4: (0, 2)
    },
    "A_1": {
        B2_7: (0, 2),
        B2_8: (0, 3),
        B2_9: (0, 1),
        B2_11: (0, 1),
        B1_14: (0, 2),
    },
    "A_2": {
        B2_7: (0, 1),
        B2_8: (0, 2),
        B2_9: (0, 1),
        B2_11: (0, 1),
        B1_14: (0, 1),
    },

    "A_3": {
        B2_9: (0, 2),
    },
    "B_1": {
        B2_11: (0, 2),
    },
    "B_2": {
        B2_11: (0, 1),
    },
}


AllMainfactors = ["A", "A_1", "A_2", "A_3", "B", "B_1", "B_2", "C", "D", "E", ]
AllDedfactors = ["Miss", "EOS"]


two_factor_list = [Y14_2_2_3_2_C]
four_factor_list = [Y14_1_1_1_3, Y14_1_2_1_3, Y14_1_2_2_4,
                    Y14_2_1_2_3, Y15_2_3_1_5, Y15_2_3_2_2, Y15_2_3_2_4]
eos_factor_list = [Y14_1_1_1_3, Y14_1_1_2_5, Y14_1_2_1_3,
                   Y14_1_2_2_4, Y14_2_1_1_5, Y15_2_3_1_4, Y15_2_3_2_4]
five_factor_list = [Y15_2_3_1_5, Y15_2_3_2_2, Y15_2_3_2_4]

metric_dic = {"accuracy": "acc", "mean_absolute_error": "mean_absolute_error"}


def get_ps_ranges():
    return ranges


def get_metric_dic():
    return metric_dic


def get_all_factors():
    return AllMainfactors, AllDedfactors


def get_four_factor_list():
    return four_factor_list


def get_eos_factor_list():
    return eos_factor_list


def get_add_weights(args):
    vs = []
    if args.cls:
        for i in range(args.part_num_main):
            vs.append(np.array([1]))
        for i in range(args.part_num_ded):
            vs.append(np.array([-1]))
    else:
        vs = get_item_score_info(args)
    W = np.array(vs)

    b = np.zeros(1)
    weight_values = [W, b]
    return weight_values


def get_item_score_info(args):
    prompt = args.prompt
    highs = []
    for i in range(args.part_num_main):
        low_ps, high_ps = ranges[args.part_names[i]][prompt]
        highs.append(np.array([high_ps]))
    for i in range(args.part_num_ded):
        highs.append(np.array([-1]))

    return highs


def get_factors(jsn):
    Mainfactors = list()
    Dedfactors = list()
    data = jsn[0].keys()
    for factor in AllMainfactors:
        if any([factor + "_Score" in k for k in data]):
            Mainfactors.append(factor)

    for factor in AllDedfactors:
        if any([factor + "_Score" in k for k in data]):
            Dedfactors.append(factor)

    return Mainfactors, Dedfactors


def prompt_check(file):
    five_factor, four_factor, two_factor, eos_factor = False, False, False, False
    prompt = ""

    for p in prompt_list:
        if p in file:
            prompt = p
            break

    if prompt in two_factor_list:
        two_factor = True
    if prompt in five_factor_list:
        five_factor = True
        four_factor = True
    if prompt in four_factor_list:
        four_factor = True
    if prompt in eos_factor_list:
        eos_factor = True
    return prompt, five_factor, four_factor, two_factor, eos_factor
