import numpy as np
import pickle
import json
from typing import List
import torch
# from util import logger as L
# from logging import getLogger
from sas import util
from itertools import islice
# logger = getLogger(__name__)


def get_special_token_id(info):
    ret = {"mask": None, "pad": None, 'sep': None, "cls": None}
    if info.BERT:
        from transformers import BertConfig, BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        ret["pad"] = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        ret["mask"] = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        ret["sep"] = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        ret["cls"] = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    else:
        ret["pad"] = info.vocab['<pad>']
        ret["mask"] = info.vocab['<unk>']
    return ret


def load_data(info, eval_size=1000000000):
    if info.BERT:
        from sas.input_data import input_data_ps_BERT as input_data
    else:
        from sas.input_data import input_data_ps as input_data

    (train_x, train_y, train_mask, train_attention, attention_flag), (dev_x, dev_y,
                                                                      dev_mask, dev_attention), (test_x, test_y, test_mask, test_attention) = input_data(info, eval_size)
    item_num = util.get_item_num(info.item, info.main_factors)
    train_y = train_y[item_num + 1]
    dev_y = dev_y[item_num + 1]
    test_y = test_y[item_num + 1]

    return (train_x, train_y, train_mask, train_attention, attention_flag), (dev_x, dev_y, dev_mask, dev_attention), (test_x, test_y, test_mask, test_attention)


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
        for line in json_data:
            yield line


def normalization(ret):
    ret = ret / np.sum(ret, axis=-1)
    return ret


def get_sas_json_data(info, mode, size=None, gold=True):
    if gold:
        if mode == 'train':
            file = info.train_dir
        elif mode == 'dev':
            file = info.dev_dir
        elif mode == 'test':
            file = info.test_dir
        else:
            raise ValueError
    else:
        file = f"{info.out_dir}_{mode}.json"
    iter = islice(load_json_file(file), 0,
                  size) if size is not None else load_json_file(file)
    for line in iter:
        yield line


def load_pickle_file(pickle_file: str) -> List:
    pickle_data = pickle.load(open(pickle_file, 'rb'))
    return pickle_data
    # justification_methods = ["attentions", "norms", "grads"]
    # for i in range(pickle_data[justification_methods[0]].shape(0)):
    #     ret = []
    #     for jm in justification_methods:
    #         ret.append(pickle_data[jm][i].tolist())
    #     yield ret


def get_explanation_pickle_data(info, mode, normalize=True):
    explanation_df = load_pickle_file(info.get_explanation_path(mode))
    if normalize:
        for method in explanation_df.columns:
            explanation_df[method] = list(
                map(normalization, explanation_df[method].to_numpy()))
        return explanation_df
    else:
        return explanation_df


def get_iteration_explanation_pickle_data(info, mode, idx=-1):
    explanation_df = get_explanation_pickle_data(info, mode, normalize=True)
    methods = explanation_df.columns
    for method in methods:
        ret = normalization(explanation_df.iloc[idx][method])
        yield method, ret


def get_sas_list(info, mode, size=None):
    gold_data = [g for g in get_sas_json_data(info, mode, size, gold=True)]
    if mode != 'train':
        pred_data = [p for p in get_sas_json_data(
            info, mode, size, gold=False)]
    else:
        pred_data = None
    if info.BERT:
        attention_key = f"C_{info.item}"
        answer_key = "Char"
    else:
        attention_key = f"{info.item}"
        answer_key = "mecab"

    sentence_list = []
    gold_attention_list = []
    gold_score_list = []
    pred_score_list = []
    id_list = []

    for i in range(len(gold_data)):
        tmp_sentence_list = gold_data[i][answer_key].split(' ')
        tmp_gold_attention_list = list(
            map(int, gold_data[i][attention_key].split()))
        if info.dmy is True:
            tmp_sentence_list.append("<dmy>")
            # gold attentionを作成
            # ダミートークンを入れる
            if sum(tmp_gold_attention_list) == 0:
                tmp_gold_attention_list.append(1)
            else:
                tmp_gold_attention_list.append(0)
        sentence_list.append(tmp_sentence_list)
        gold_attention_list.append(tmp_gold_attention_list)
        gold_score_list.append(gold_data[i][f"{info.item}_Score"])
        pred_score_list.append(
            pred_data[i][f"{info.item}_score"] if pred_data else None)
        id_list.append(gold_data[i]["id"])

    return sentence_list, gold_attention_list, gold_score_list, pred_score_list, id_list


def get_eraser_list(info, mode, size=None):
    gold_data = [g for g in get_sas_json_data(info, mode, size, gold=True)]
    if mode != 'train':
        pred_data = [p for p in get_sas_json_data(
            info, mode, size, gold=False)]
    else:
        pred_data = None
    if info.BERT:
        # attention_key = f"bert_{info.item}"
        if info.char:
            attention_key = f"C_{info.item}"
        else:
            attention_key = f"bert_{info.item}"
        answer_key = "bert"
    else:
        attention_key = f"{info.item}"
        answer_key = "mecab"

    sentence_list = []
    gold_attention_list = []
    gold_score_list = []
    pred_score_list = []
    id_list = []

    for i in range(len(gold_data)):
        tmp_sentence_list = gold_data[i][answer_key].split(' ')
        tmp_gold_attention_list = list(
            map(int, gold_data[i][attention_key].split()))
        if info.dmy is True:
            tmp_sentence_list.append("<dmy>")
            # gold attentionを作成
            # ダミートークンを入れる
            if sum(tmp_gold_attention_list) == 0:
                tmp_gold_attention_list.append(1)
            else:
                tmp_gold_attention_list.append(0)
        sentence_list.append(tmp_sentence_list)
        gold_attention_list.append(tmp_gold_attention_list)
        gold_score_list.append(gold_data[i][f"{info.item}_Score"])
        pred_score_list.append(
            pred_data[i][f"{info.item}_score"] if pred_data else None)
        id_list.append(gold_data[i]["id"])
    return sentence_list, gold_attention_list, gold_score_list, pred_score_list, id_list


def get_sas_list_for_xlsx(info, mode, size=None):
    sas_data = get_sas_list(info, mode, size)
    for sentence_list, gold_attention_list, gold_score, pred_score, id in zip(sas_data[0], sas_data[1], sas_data[2], sas_data[3], sas_data[4]):
        yield sentence_list, gold_attention_list, gold_score, pred_score, id


def get_moview_review_list_for_xlsx(info, mode, size=None):
    sas_data = get_eraser_list(info, mode, size)
    for sentence_list, gold_attention_list, gold_score, pred_score, id in zip(sas_data[0], sas_data[1], sas_data[2], sas_data[3], sas_data[4]):
        yield sentence_list, gold_attention_list, gold_score, pred_score, id


def to_descrete_by_threshold(justification_cue, threshold):
    ret = []
    max_justi = max(justification_cue)
    for i in range(len(justification_cue)):
        if 0 < justification_cue[i] and max_justi - justification_cue[i] < threshold:
            ret.append(1)
        else:
            ret.append(0)
    return ret


def exclude_zero_score_answer(gold_justification_list: List[float], pred_justification_list: List[float], gold_score_list: List[int]) -> (List[float], List[float]):
    gold_ret_list = []
    pred_ret_list = []
    for gj, pj, gs in zip(gold_justification_list, pred_justification_list, gold_score_list):
        if gs != 0:
            gold_ret_list.append(gj)
            pred_ret_list.append(pj)
    return gold_ret_list, pred_ret_list


def calc_auprc(gold_justification_list: List[float], pred_justification_list: List[float]) -> float:
    auprc_list = []
    for gjl, pjl in zip(gold_justification_list, pred_justification_list):
        precision, recall, thresholds = precision_recall_curve(
            gjl, pjl[:len(gjl)])
        auprc = 0.0
        pre_r, pre_p = 1, 0
        for r, p in zip(recall, precision):
            auprc += (pre_p + p) * (pre_r - r) / 2
            pre_p, pre_r = p, r
        auprc_list.append(auprc)
        return np.mean(auprc_list)


def load_model(info):
    from model import BertForScoring, BertForPartScoring, BertForPartScoringWithLSTM, BiRnnModel, BiRnnModelForItemScoring
    # モデルのアーキテクチャを取得
    if info.BERT:
        from transformers import BertConfig
        # # config  = BertConfig.from_pretrained(info.config)
        # if info.char:
        #     config = BertConfig.from_pretrained(
        #         "cl-tohoku/bert-base-japanese-char-whole-word-masking")
        # else:
        #     config = BertConfig.from_pretrained(
        #         "cl-tohoku/bert-base-japanese-whole-word-masking")
        # info.emb_dim = config.hidden_size
        # model = BiRnnModelForItemScoring(info, config=config)
        if info.char and "char" not in info.config:
            raise ValueError(f"{info.config} is not char model")
        elif not info.char and "char" in info.config:
            raise ValueError(f"{info.config} is char model")
        model = BiRnnModelForItemScoringERASER(info, config=info.config)
        model.freeze_bert_pram()
    else:
        if info.implementation == 'asazuma':
            from model_asazuma import LSTMAttention
            model = LSTMAttention(info)
        else:
            model = BiRnnModelForItemScoring(info)

    # モデルをロード
    model.load_state_dict(torch.load(
        info.model_path, map_location=torch.device(info.device))["model_state_dict"])
    model = model.to(info.device)

    model.print_model_info()
    # logger.info(f"Load pretrained model from {info.model_path}")

    return model


def load_model_eraser(info):
    from model_eraser import BiRnnModelForItemScoringERASER, BertFinetuningForItemScoringERASER
    # モデルのアーキテクチャを取得
    if info.BERT:
        from transformers import BertConfig
        # config  = BertConfig.from_pretrained(info.config)
        # if info.char:
        #     config = BertConfig.from_pretrained(
        #         "bert-base-uncased")
        # else:
        #     config = BertConfig.from_pretrained(
        #         "bert-base-uncased")
        if info.char and "char" not in info.config:
            raise ValueError(f"{info.config} is not char model")
        elif not info.char and "char" in info.config:
            raise ValueError(f"{info.config} is char model")

        if info.implementation == 'no_lstm_bert':
            model = BertFinetuningForItemScoringERASER(
                info, config=info.config)
        elif info.implementation == 'funayama':
            model = BiRnnModelForItemScoringERASER(info, config=info.config)
        else:
            raise ValueError(f"we don't have {info.implementation} model")
        model.freeze_bert_pram()
    else:
        if info.implementation == 'asazuma':
            from model_asazuma import LSTMAttention
            model = LSTMAttention(info)
        else:
            model = BiRnnModelForItemScoringERASER(info)

    # モデルをロード
    model.load_state_dict(torch.load(
        info.model_path, map_location=torch.device(info.device))["model_state_dict"])
    model = model.to(info.device)

    return model


def load_instance_based_model(info):
    from model import BertForPartScoringWithLSTM, BiRnnModel, BiRnnModelForItemScoringWithMetric
    # モデルのアーキテクチャを取得
    if info.BERT:
        from transformers import BertConfig
        # config  = BertConfig.from_pretrained(info.config)
        if info.char:
            config = BertConfig.from_pretrained(
                "cl-tohoku/bert-base-japanese-char-whole-word-masking")
        else:
            config = BertConfig.from_pretrained(
                "cl-tohoku/bert-base-japanese-whole-word-masking")
        info.emb_dim = config.hidden_size
        model = BiRnnModelForItemScoringWithMetric(
            info, config=config)
        model.freeze_bert_pram()
    else:
        model = BiRnnModelForItemScoringWithMetric(
            info)
    # モデルをロード
    model.load_state_dict(torch.load(
        info.model_path, map_location=torch.device(info.device))["model_state_dict"])
    model = model.to(info.device)

    # logger.info(f"Load pretrained model from {info.model_path}")

    return model
