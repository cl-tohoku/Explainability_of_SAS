from logging import getLogger
import torch
import codecs
from collections import defaultdict
import sys
from itertools import islice

logger = getLogger(__name__)


def input_data(info):
    data_path = (info.train_dir, info.dev_dir, info.test_dir)
    from transformers import BertJapaneseTokenizer

    import torch
    import part_scoring_info as psi

    if info.char:
        # tokenizer = MecabCharacterBertTokenizer(vocab_file=info.vocab_dir)
        tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-char-whole-word-masking")
    else:
        # tokenizer = MecabBertTokenizer(vocab_file=info.vocab_dir)
        tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking")

    d_names = ("train", "dev", "test")
    data = dict()

    overal_maxlen = 0
    p_maxlen = []
    for name, path in zip(d_names, data_path):
        answers, score = read_data(path, info.labels, info.low, info.high)
        if isinstance(score, str):
            score = int(score)
        ids = [tokenizer.encode(sent, add_special_tokens=True)
               for sent in answers]

        p_maxlen.append(max([len(id) for id in ids]))
        data[f"{name}_ids"] = ids
        data[f"{name}_y"] = torch.tensor(score)

    overal_maxlen = max(p_maxlen)
    info.overal_maxlen = overal_maxlen
    for name in d_names:
        data[f"{name}_mask"] = torch.LongTensor(
            padding(data[f"{name}_ids"], overal_maxlen))
        data[f"{name}_x"] = torch.LongTensor(data[f"{name}_ids"])

    return (data["train_x"], data["train_y"], data["train_mask"]), (data["dev_x"], data["dev_y"], data["dev_mask"]), (data["test_x"], data["test_y"], data["test_mask"])

#
# def input_data_ps_BERT(info):
#     data_path = (info.train_dir, info.dev_dir, info.test_dir)
#     from bert_scripts.tokenization import MecabBertTokenizer, MecabCharacterBertTokenizer
#     import torch
#     import part_scoring_info as psi
#     import json
#     from transformers import BertJapaneseTokenizer
#
#     if info.char:
#         # tokenizer = MecabCharacterBertTokenizer(vocab_file=info.vocab_dir)
#         tokenizer = BertJapaneseTokenizer.from_pretrained(
#             "cl-tohoku/bert-base-japanese-char-whole-word-masking")
#     else:
#         # tokenizer = MecabBertTokenizer(vocab_file=info.vocab_dir)
#         tokenizer = BertJapaneseTokenizer.from_pretrained(
#             "cl-tohoku/bert-base-japanese-whole-word-masking")
#     d_names = ("train", "dev", "test")
#     data = dict()
#
#     overal_maxlen = 0
#     p_maxlen = []
#
#     info.main_factors, info.ded_factors = psi.get_factors(
#         json.load(codecs.open(info.train_dir,encoding='utf-8')))
#     factors = psi.get_all_factors()
#     ranges = psi.get_ps_ranges()
#     prompt = psi.prompt_check(info.train_dir)[0]
#     info.store_part_scoring_info()
#
#     for name, path in zip(d_names, data_path):
#         answers, scores_ps, attention_ps = read_data_ps_BERT(
#             path, info)
#
#         ids = [tokenizer.encode(sent, add_special_tokens=True)
#                for sent in answers]
#
#         maxlen = max([len(id) for id in ids])
#         p_maxlen.append(maxlen)
#         data[f"{name}_ids"] = ids
#         data[f"{name}_y"] = torch.tensor(scores_ps)
#         if name == "train":
#             train_maxlen = maxlen
#             data[f"{name}_attention"] = attention_ps
#
#     overal_maxlen = max(p_maxlen)
#     info.overal_maxlen = overal_maxlen
#
#     for name in d_names:
#         data[f"{name}_mask"] = torch.LongTensor(padding(data[f"{name}_ids"], overal_maxlen))
#         if name == "train":
#             padding_attn(data[f"{name}_attention"], overal_maxlen)
#             data[f"{name}_satt"] = torch.FloatTensor(data[f"{name}_attention"])
#         data[f"{name}_x"] = torch.LongTensor(data[f"{name}_ids"])
#
#     return (data["train_x"], data["train_y"], data["train_mask"], data["train_satt"]), (data["dev_x"], data["dev_y"], data["dev_mask"]), (data["test_x"], data["test_y"], data["test_mask"])


def input_data_ps_BERT(info, eval_size=None):
    data_path = (info.train_dir, info.dev_dir, info.test_dir)
    from bert_scripts.tokenization import MecabBertTokenizer, MecabCharacterBertTokenizer
    import torch
    import part_scoring_info as psi
    import json
    from transformers import BertJapaneseTokenizer, BertTokenizer

    # if info.char:
    #     # tokenizer = MecabCharacterBertTokenizer(vocab_file=info.vocab_dir)
    #     tokenizer = BertJapaneseTokenizer.from_pretrained(
    #         "cl-tohoku/bert-base-japanese-char-whole-word-masking")
    # else:
    #     # tokenizer = MecabBertTokenizer(vocab_file=info.vocab_dir)
    #     tokenizer = BertTokenizer.from_pretrained( "bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(info.config)
    info.vocab = dict(tokenizer.vocab)
    d_names = ("train", "dev", "test")
    data = dict()

    overal_maxlen = 0
    p_maxlen = []

    info.main_factors, info.ded_factors = psi.get_factors(
        json.load(codecs.open(info.train_dir, encoding='utf-8')))
    factors = psi.get_all_factors()
    ranges = psi.get_ps_ranges()
    prompt = psi.prompt_check(info.train_dir)[0]
    info.store_part_scoring_info()

    for name, path in zip(d_names, data_path):
        answers, scores_ps, attention_ps = read_data_ps_BERT(
            path, info, eval_size)
        # ids = [tokenizer.encode(sent, add_special_tokens=True)
        #        for sent in answers]
        ids = []
        batch_len = 300
        for sent in answers:
            id = tokenizer.convert_tokens_to_ids(["[CLS]"])
            if info.char:
                sent_split = list(sent)
            else:
                sent_split = sent.split(' ')
            for batch_index in range(0, len(sent_split), batch_len):
                # id.extend(tokenizer(' '.join(sent_split[batch_index:batch_index+batch_len]), return_tensors='pt',padding=True, truncation=True)["input_ids"][0][1:-1])
                id.extend(tokenizer.convert_tokens_to_ids(
                    sent_split[batch_index:batch_index + batch_len]))
            id.append(tokenizer.convert_tokens_to_ids(["[SEP]"])[0])
            ids.append(id)
        maxlen = max([len(id) for id in ids])
        p_maxlen.append(maxlen)
        data[f"{name}_ids"] = ids
        data[f"{name}_y"] = torch.tensor(scores_ps)
        data[f"{name}_attention"] = attention_ps

    overal_maxlen = max(p_maxlen)
    info.overal_maxlen = overal_maxlen

    for name in d_names:
        data[f"{name}_mask"] = torch.LongTensor(
            padding(data[f"{name}_ids"], overal_maxlen))
        padding_attn(data[f"{name}_attention"], overal_maxlen)
        data[f"{name}_satt"] = torch.FloatTensor(data[f"{name}_attention"])
        data[f"{name}_x"] = torch.LongTensor(data[f"{name}_ids"])

    if info.attention_train_size is None:
        logger.info(
            f"Make attention train data size the same as the scoring train size:{data['train_x'].shape[0]}")
        info.attention_train_size = data["train_x"].shape[0]
    elif data["train_x"].shape[0] < info.attention_train_size:
        logger.warning(
            f"attention train data is larger than row data:{info.attention_train_size}->{data['train_x'].shape[0]}")
        info.attention_train_size = data["train_x"].shape[0]

    attention_flag = torch.tensor([[True for _ in range(info.attention_train_size)] + [False for _ in range(
        data["train_x"].shape[0] - info.attention_train_size)] for _ in range(data["train_y"].shape[0])])

    # # 0点答案でアテンションを学習させない時
    if info.dmy is False:
        attention_flag = torch.logical_and(
            data["train_y"] != 0, attention_flag)
        assert not torch.any(torch.logical_and(
            data["train_y"] == 0, attention_flag)), f"{data['train_y']},\n{attention_flag}"
    return (data["train_x"], data["train_y"], data["train_mask"], data["train_satt"], attention_flag), (data["dev_x"], data["dev_y"], data["dev_mask"], data["dev_satt"]), (data["test_x"], data["test_y"], data["test_mask"], data["test_satt"])


#
# def input_data_ps(info):
#     data_path = (info.train_dir, info.dev_dir, info.test_dir)
#     import torch
#     import part_scoring_info as psi
#     import json
#
#     if info.char:
#         e_field = "Char"
#     else:
#         e_field = "mecab"
#
#     if info.emb_path != None:
#         vocab = create_vocab_from_pretrained_emb(
#             (info.train_dir,info.dev_dir,info.test_dir), emb_path=info.emb_path, field=e_field)
#     else:
#         vocab = create_vocab(
#             info.train_dir, vocab_size=info.vocab_size, field=e_field)
#     #
#     # vocab = create_vocab(
#     #             info.train_dir, vocab_size=info.vocab_size,field=e_field)
#
#     info.vocab = vocab
#
#     d_names = ("train", "dev", "test")
#     data = dict()
#
#     overal_maxlen = 0
#     p_maxlen = []
#
#     info.main_factors, info.ded_factors = psi.get_factors(
#         json.load(codecs.open(info.train_dir,encoding='utf-8')))
#     factors = psi.get_all_factors()
#     ranges = psi.get_ps_ranges()
#     prompt = psi.prompt_check(info.train_dir)[0]
#     info.store_part_scoring_info()
#
#     unk_set = defaultdict(set)
#     for name, path in zip(d_names, data_path):
#         if info.reg:
#             answers, scores_ps, attention_ps = read_data_ps_for_reg(
#                 path, info, e_field)
#         else:
#             answers, scores_ps, attention_ps = read_data_ps(
#                 path, info, e_field)
#
#         ids = []
#
#         for answer in answers:
#             encoded_text = []
#             for sent in answer.split(" "):
#                 id = vocab.get(sent,1)
#                 if id == 1:
#                     unk_set[name].add(sent)
#                 encoded_text.append(id)
#             ids.append(encoded_text)
#
#         maxlen = max([len(id) for id in ids])
#         p_maxlen.append(maxlen)
#         data[f"{name}_ids"] = ids
#         data[f"{name}_y"] = torch.tensor(scores_ps)
#         if name == "train":
#             train_maxlen = maxlen
#             data[f"{name}_attention"] = attention_ps
#         logger.info(f"{name}: {len(unk_set[name])} unique words replaced with <UNK>.")
#     overal_maxlen = max(p_maxlen)
#     info.overal_maxlen = overal_maxlen
#
#     for name in d_names:
#         data[f"{name}_mask"] = torch.LongTensor(padding(data[f"{name}_ids"], overal_maxlen))
#         if name == "train":
#             padding_attn(data[f"{name}_attention"], overal_maxlen)
#             data[f"{name}_satt"] = torch.FloatTensor(data[f"{name}_attention"])
#         data[f"{name}_x"] = torch.LongTensor(data[f"{name}_ids"])
#
#     return (data["train_x"], data["train_y"], data["train_mask"], data["train_satt"]), (data["dev_x"], data["dev_y"], data["dev_mask"]), (data["test_x"], data["test_y"], data["test_mask"])

def input_data_ps(info, eval_size=None):
    data_path = (info.train_dir, info.dev_dir, info.test_dir)
    import torch
    import part_scoring_info as psi
    import json

    if info.char:
        e_field = "Char"
    else:
        e_field = "mecab"

    if info.emb_path != None:
        vocab = create_vocab_from_pretrained_emb(
            (info.train_dir, info.dev_dir, info.test_dir), emb_path=info.emb_path, field=e_field)
    else:
        vocab = create_vocab(
            info.train_dir, vocab_size=info.vocab_size, field=e_field)
    #
    # vocab = create_vocab(
    #             info.train_dir, vocab_size=info.vocab_size,field=e_field)

    info.vocab = vocab

    d_names = ("train", "dev", "test")
    data = dict()

    overal_maxlen = 0
    p_maxlen = []

    info.main_factors, info.ded_factors = psi.get_factors(
        json.load(codecs.open(info.train_dir, encoding='utf-8')))
    factors = psi.get_all_factors()
    ranges = psi.get_ps_ranges()
    prompt = psi.prompt_check(info.train_dir)[0]
    info.store_part_scoring_info()

    unk_set = defaultdict(set)
    for name, path in zip(d_names, data_path):
        if info.reg:
            answers, scores_ps, attention_ps = read_data_ps_for_reg(
                path, info, eval_size, e_field)
        else:
            answers, scores_ps, attention_ps = read_data_ps(
                path, info, eval_size, e_field)

        ids = []

        for answer in answers:
            encoded_text = []
            for sent in answer.split(" "):
                id = vocab.get(sent, 1)
                if id == 1:
                    unk_set[name].add(sent)
                encoded_text.append(id)
            ids.append(encoded_text)

        maxlen = max([len(id) for id in ids])
        p_maxlen.append(maxlen)
        data[f"{name}_ids"] = ids
        data[f"{name}_y"] = torch.tensor(scores_ps)

        data[f"{name}_attention"] = attention_ps
        logger.info(
            f"{name}: {len(unk_set[name])} unique words replaced with <UNK>.")
    overal_maxlen = max(p_maxlen)
    info.overal_maxlen = overal_maxlen

    for name in d_names:
        data[f"{name}_mask"] = torch.LongTensor(
            padding(data[f"{name}_ids"], overal_maxlen))
        padding_attn(data[f"{name}_attention"], overal_maxlen)
        data[f"{name}_satt"] = torch.FloatTensor(data[f"{name}_attention"])
        data[f"{name}_x"] = torch.LongTensor(data[f"{name}_ids"])

    if info.attention_train_size is None:
        logger.info(
            f"Make attention train data size the same as the scoring train size:{data['train_x'].shape[0]}")
        info.attention_train_size = data["train_x"].shape[0]
    elif data["train_x"].shape[0] < info.attention_train_size:
        logger.warning(
            f"attention train data is larger than row data:{info.attention_train_size}->{data['train_x'].shape[0]}")
        info.attention_train_size = data["train_x"].shape[0]

    attention_flag = torch.tensor([[True for _ in range(info.attention_train_size)] + [False for _ in range(
        data["train_x"].shape[0] - info.attention_train_size)] for _ in range(data["train_y"].shape[0])])

    # # 0点答案でアテンションを学習させない時
    if info.dmy is False:
        attention_flag = torch.logical_and(
            data["train_y"] != 0, attention_flag)
        assert not torch.any(torch.logical_and(
            data["train_y"] == 0, attention_flag)), f"{data['train_y']},\n{attention_flag}"
    return (data["train_x"], data["train_y"], data["train_mask"], data["train_satt"], attention_flag), (data["dev_x"], data["dev_y"], data["dev_mask"], data["dev_satt"]), (data["test_x"], data["test_y"], data["test_mask"], data["test_satt"])


def get_answers(info):
    import json

    if info.char:
        e_field = "Char"
    else:
        e_field = "mecab"

    data_path = (info.train_dir, info.dev_dir, info.test_dir)
    d_names = ["train", "dev", "test"]
    answers = defaultdict(list)
    for name, path in zip(d_names, data_path):
        answer, scores_ps, attention_ps = read_data_ps_for_reg(
            path, info, e_field)
        answers[name] = answer

    return answers


def get_justification_word_set(info):
    import json

    justification_word_set = set()
    if info.char:
        e_field = "Char"
    else:
        e_field = "mecab"

    from sas import util
    item_num = util.get_item_num(info.item, info.main_factors)

    data_path = (info.train_dir, info.dev_dir, info.test_dir)
    d_names = ["train", "dev", "test"]

    for name, path in zip(d_names, data_path):
        answer, scores_ps, attention_ps = read_data_ps_for_reg(
            path, info, e_field)
        # answers[name] = answer
        for i in range(len(answer)):
            # tokens = [t if t != '[space]' else ' ' for t in answer[i].replace(
            #     "   ", " [space] ").split()]
            tokens = answer[i].split(' ')
            assert len(tokens) == len(attention_ps[item_num][i]
                                      ), f"{answer[i]} {len(tokens)}, {attention_ps[item_num][i]} {len(attention_ps[item_num][i])}"
            for token, att in zip(tokens, attention_ps[item_num][i]):
                if 0 < att:
                    justification_word_set.add(token)

    return justification_word_set


def input_data_ps_regression(info):
    data_path = (info.train_dir, info.dev_dir, info.test_dir)
    import torch
    import part_scoring_info as psi
    import json

    if info.char:
        e_field = "Char"
    else:
        e_field = "mecab"

    if info.emb_path != None:
        vocab = create_vocab_from_pretrained_emb(
            (info.train_dir, info.dev_dir, info.test_dir), emb_path=info.emb_path, field=e_field)
    else:
        vocab = create_vocab(
            info.train_dir, vocab_size=info.vocab_size, field=e_field)
    #
    # vocab = create_vocab(
    #             info.train_dir, vocab_size=info.vocab_size,field=e_field)

    info.vocab = vocab

    d_names = ("train", "dev", "test")
    data = dict()

    overal_maxlen = 0
    p_maxlen = []

    info.main_factors, info.ded_factors = psi.get_factors(
        json.load(codecs.open(info.train_dir, encoding='utf-8')))
    factors = psi.get_all_factors()
    ranges = psi.get_ps_ranges()
    prompt = psi.prompt_check(info.train_dir)[0]

    info.ps_labels = [ranges[factor][prompt] for factor in info.main_factors]

    unk_set = defaultdict(set)
    for name, path in zip(d_names, data_path):
        answers, scores_ps, attention_ps = read_data_ps(
            path, info, e_field)

        ids = []

        for answer in answers:
            encoded_text = []
            for sent in answer.split(" "):
                id = vocab.get(sent, 1)
                if id == 1:
                    unk_set[name].add(sent)
                encoded_text.append(id)
            ids.append(encoded_text)

        maxlen = max([len(id) for id in ids])
        p_maxlen.append(maxlen)
        data[f"{name}_ids"] = ids
        data[f"{name}_y"] = torch.tensor(scores_ps)
        if name == "train":
            train_maxlen = maxlen
            data[f"{name}_attention"] = attention_ps
        logger.info(
            f"{name}: {len(unk_set[name])} unique words replaced with <UNK>.")
    overal_maxlen = max(p_maxlen)
    info.overal_maxlen = overal_maxlen

    for name in d_names:
        data[f"{name}_mask"] = torch.LongTensor(
            padding(data[f"{name}_ids"], overal_maxlen))
        if name == "train":
            padding_attn(data[f"{name}_attention"], overal_maxlen)
            data[f"{name}_satt"] = torch.FloatTensor(data[f"{name}_attention"])
        data[f"{name}_x"] = torch.LongTensor(data[f"{name}_ids"])

    return (data["train_x"], data["train_y"], data["train_mask"], data["train_satt"]), (data["dev_x"], data["dev_y"], data["dev_mask"]), (data["test_x"], data["test_y"], data["test_mask"])


def read_data(data_path, labels, low, high):
    import json
    jsn = json.load(codecs.open(data_path, "r", encoding='utf-8'))

    scores = list()
    ans = list()

    for data in jsn:
        text = data["mecab"].replace(" ", "")

        score = int(data["score"])
        if score <= 0:
            score = 0
        assert score <= high and score >= low, logger.error(
            f"Score is invalid ->{text}:{low} <= {score} <={high}")

        scores.append(score)
        ans.append(text)

    return ans, scores


def read_data_ps_BERT(data_path, info, eval_size):
    import json
    jsn = json.load(codecs.open(data_path, "r", encoding='utf-8'))

    scores_ps = list()
    ans = list()

    score_dict = {"score": []}
    attention = {}

    for factor in info.main_factors:
        score_dict[factor] = []
        if info.char:
            attention['C_' + factor] = []
        else:
            # attention[factor] = []
            attention[f"bert_{factor}"] = []
    for factor in info.ded_factors:
        score_dict[factor] = []

    factors = info.main_factors + info.ded_factors

    for data in jsn[:eval_size]:
        if info.char:
            text = data["mecab"].replace(" ", "")
            if info.dmy:
                text += '◇'
        else:
            text = data["bert"]
            if info.dmy:
                text += ' ＃'
                assert False, "charの時は#じゃなくて◇をダミートークンとしている"
        score = data["score"]
        if score <= 0:
            score = 0
        score_dict["score"].append(score)

        ans.append(text)

        for i, factor in enumerate(factors):
            k = factor + "_Score"
            score = data[k]
            if i < len(info.main_factors):
                assert score <= info.ps_labels[i][1] and score >= info.ps_labels[
                    i][0], logger.error(f"Score is invalid ->{text}:{factor}:{score}:{info.ps_labels[i]}")

                if info.char:
                    k = "C_" + factor
                    if score == 0 and info.no_use_fuka_ann:
                        attn = [0.0
                                for attn in data[k].split()]
                    else:
                        attn = [float(attn)
                                for attn in data[k].split()]

                    if info.dmy:
                        if not any(attn):
                            attn.append(1.0)
                        else:
                            attn.append(0.0)

                    attn_sum = sum(attn)

                    if attn_sum != 0:
                        attention["C_" + factor].append([num / attn_sum
                                                         for num in attn])
                    else:
                        attention["C_" + factor].append(attn)

                    '''
                    if attn_sum != 0:
                        attention["C_" + factor].append(attn)
                    else:
                        attention["C_" + factor].append(attn)
                    '''

                else:
                    k = "bert_" + factor
                    if score == 0 and info.no_use_fuka_ann:
                        attn = [0.0 for attn in data[k].split()]
                    else:
                        attn = [float(attn) for attn in data[k].split()]

                    if info.dmy:
                        if not any(attn):
                            attn.append(1.0)
                        else:
                            attn.append(0.0)

                    attn_sum = sum(attn)

                    if attn_sum != 0:
                        attention[k].append([num / attn_sum
                                             for num in attn])
                    else:
                        attention[k].append(attn)
            score_dict[factor].append(score)

    scores = [ps_scores for ps_scores in islice(
        score_dict.values(), 0, eval_size)]
    attentions = [ps_attention for ps_attention in islice(
        attention.values(), 0, eval_size)]

    # attn_len = len(attentions[0][0])
    # for attn in attentions[0]:
    #     assert attn_len == len(attn), f"{attn_len} {len(attn)}"
    #     print(len(attn))
    return ans, scores, attentions


def read_data_ps(data_path, info, eval_size, e_field="mecab"):
    import json
    jsn = json.load(codecs.open(data_path, "r", encoding='utf-8'))

    scores_ps = list()
    ans = list()

    score_dict = {"score": []}
    attention = {}

    for factor in info.main_factors:
        score_dict[factor] = []
        attention[factor] = []
    for factor in info.ded_factors:
        score_dict[factor] = []

    factors = info.main_factors + info.ded_factors

    for data in jsn[:eval_size]:
        text = data[e_field]
        if info.dmy:
            text += " <dmy>"
        score = data["score"]
        if score <= 0:
            score = 0
        score_dict["score"].append(score)

        ans.append(text)

        for i, factor in enumerate(factors):
            k = factor + "_Score"
            score = data[k]
            if i < len(info.main_factors):
                assert score <= info.ps_labels[i][1] and score >= info.ps_labels[i][
                    0], f"{score}, {info.ps_labels[i][1]}, {info.ps_labels[i][0]}"
                # logger.error(f"Score is invalid ->{text}:{factor}:{score}:{info.ps_labels[i]}")

                if info.char:
                    k = "C_" + factor
                    if score == 0 and info.no_use_fuka_ann:
                        attn = [0.0
                                for attn in data[k].split()]
                    else:
                        attn = [float(attn)
                                for attn in data[k].split()]

                    if info.dmy:
                        if not any(attn):
                            attn.append(1.0)
                        else:
                            attn.append(0.0)

                    attn_sum = sum(attn)

                    if attn_sum != 0:
                        attention[factor].append([num / attn_sum
                                                  for num in attn])
                    else:
                        attention[factor].append(attn)

                else:
                    if score == 0 and info.no_use_fuka_ann:
                        attn = [0.0 for attn in data[factor].split()]
                    else:
                        attn = [float(attn) for attn in data[factor].split()]

                    if info.dmy:
                        if not any(attn):
                            attn.append(1.0)
                        else:
                            attn.append(0.0)

                    attn_sum = sum(attn)

                    if attn_sum != 0:
                        attention[factor].append([num / attn_sum
                                                  for num in attn])
                    else:
                        attention[factor].append(attn)

            score_dict[factor].append(score)

    scores = [ps_scores for ps_scores in islice(
        score_dict.values(), 0, eval_size)]
    attentions = [ps_attention for ps_attention in islice(
        attention.values(), 0, eval_size)]

    return ans, scores, attentions


def read_data_ps_for_reg(data_path, info, eval_size, e_field="mecab"):
    import json
    jsn = json.load(codecs.open(data_path, "r", encoding='utf-8'))

    scores_ps = list()
    ans = list()

    score_dict = {"score": []}
    attention = {}

    for factor in info.main_factors:
        score_dict[factor] = []
        attention[factor] = []
    for factor in info.ded_factors:
        score_dict[factor] = []

    factors = info.main_factors + info.ded_factors

    for data in jsn[:eval_size]:
        text = data[e_field]
        if info.dmy:
            text += " <dmy>"
        score = data["score"]
        if score <= 0:
            score = 0
        score_dict["score"].append(score)

        ans.append(text)

        for i, factor in enumerate(factors):
            k = factor + "_Score"

            score = data[k] / info.ps_labels[i][1]
            if i < len(info.main_factors):
                assert score <= info.ps_labels[i][1] and score >= info.ps_labels[i][0], logger.error(
                    f"Score is invalid ->{text}:{factor}:{score}:{info.ps_labels[i]}")

                if info.char:
                    k = "C_" + factor
                    if score == 0 and info.no_use_fuka_ann:
                        attn = [0.0
                                for attn in data[k].split()]
                    else:
                        attn = [float(attn)
                                for attn in data[k].split()]

                    if info.dmy:
                        if not any(attn):
                            attn.append(1.0)
                        else:
                            attn.append(0.0)

                    attn_sum = sum(attn)

                    if attn_sum != 0:
                        attention[factor].append([num / attn_sum
                                                  for num in attn])
                    else:
                        attention[factor].append(attn)

                else:
                    attn = [float(attn) for attn in data[factor].split()]

                    if info.dmy:
                        if not any(attn):
                            attn.append(1.0)
                        else:
                            attn.append(0.0)

                    attn_sum = sum(attn)

                    if attn_sum != 0:
                        attention[factor].append([num / attn_sum
                                                  for num in attn])
                    else:
                        attention[factor].append(attn)

            score_dict[factor].append(score)

    scores = [ps_scores for ps_scores in islice(
        score_dict.values(), 0, eval_size)]
    attentions = [ps_attention for ps_attention in islice(
        attention.values(), 0, eval_size)]

    return ans, scores, attentions


def padding(ids, overal_maxlen):
    input_mask = list()
    for id in ids:
        pads = [0] * (overal_maxlen - len(id))
        id.extend(pads)
        input_mask.append([float(i > 0) for i in id])

    return input_mask


def padding_attn(attentions, overal_maxlen):

    input_mask = list()
    for attention_ps in attentions:
        for attn in attention_ps:
            pads = [0] * (overal_maxlen - len(attn))
            attn.extend(pads)


def get_prompt(info):
    import part_scoring_info

    prompt, five_factor, four_factor, two_factor, eos_factor = part_scoring_info.prompt_check(
        info.train_dir)

    return prompt


def convert_to_model_friendlly_score(info, data_y):

    data_y = data_y.type(torch.float32)
    data_y_model_friendly = data_y / info.high

    return data_y_model_friendly


def input_data_inference_BERT(info, text):
    from bert_scripts.tokenization import MecabBertTokenizer, MecabCharacterBertTokenizer
    import torch

    if info.char:
        tokenizer = MecabCharacterBertTokenizer(vocab_file=info.vocab_dir)
    else:
        tokenizer = MecabBertTokenizer(vocab_file=info.vocab_dir)
    ids = tokenizer.encode(text, add_special_tokens=True)
    id = torch.LongTensor([ids])
    id.requires_grad = False

    return id


def input_data_inference(info, text):
    from bert_scripts.tokenization import MecabBertTokenizer, MecabCharacterBertTokenizer
    import torch
    vocab = info.vocab
    overall_maxlen = info.overal_maxlen
    ids = [0] * overall_maxlen
    attn_mask = [0] * overall_maxlen
    for i, word in enumerate(text):
        ids[i] = vocab.get(word, 1)
        attn_mask[i] = 1

    id = torch.LongTensor([ids])
    attn_mask_tensor = torch.LongTensor([attn_mask])
    id.requires_grad = False

    return id, attn_mask_tensor


def create_vocab(file_path, vocab_size, field="mecab"):
    import codecs
    import json
    logger.info('Creating vocabulary from: ' + file_path)
    total_words, unique_words = 0, 0
    word_freqs = {}

    f = codecs.open(file_path, 'r', encoding="utf-8")
    jsonData = json.load(f)
    f.close()

    for line in jsonData:
        content = line[field]
        words = content.split(" ")
        for word in words:
            try:
                word_freqs[word] += 1
            except KeyError:
                unique_words += 1
                word_freqs[word] = 1
            total_words += 1
    logger.info('  %i total words, %i unique words' %
                (total_words, unique_words))
    import operator
    sorted_word_freqs = sorted(
        word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2,
             '<dmy>': 3, '<rand>': 4, '<zero>': 5}
    vcb_len = len(vocab)
    index = vcb_len
    for word, freq in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1
    return vocab


def create_vocab_from_pretrained_emb(file_path, emb_path, field="mecab"):
    import codecs
    import json
    d_names = ("train", "dev", "test")

    logger.info("Create vocabulary from %s" % (emb_path))
    total_words, unique_words = 0, 0
    candidate_words = set()
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2,
             '<dmy>': 3, '<rand>': 4, '<zero>': 5}

    # f = codecs.open(file_path[0], 'r', encoding="utf-8")
    # jsonData = json.load(f)
    # f.close()
    # for line in jsonData:
    #     content = line[field]
    #     words = content.split(" ")
    #     for word in words:
    #         if not (word in vocab):
    #             unique_words += 1
    #             vocab[word] = len(vocab)
    #         total_words += 1

    for path, name in zip(file_path, d_names):
        f = codecs.open(path, 'r', encoding="utf-8")
        jsonData = json.load(f)
        f.close()
        for line in jsonData:
            content = line[field]
            words = content.split(" ")
            for word in words:
                if not (word in candidate_words):
                    unique_words += 1
                    candidate_words.add(word)
                total_words += 1
    logger.info('  %i total words, %i unique words' %
                (total_words, unique_words))

    with codecs.open(emb_path, "r", encoding='utf-8') as fi:
        for i, line in enumerate(fi):
            if i == 0:
                continue
            line_list = line.strip().split(" ", 1)
            word = line_list[0]
            if word in candidate_words:
                vocab[word] = len(vocab)

    return vocab
