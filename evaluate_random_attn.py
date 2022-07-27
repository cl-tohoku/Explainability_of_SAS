'''
attention weightをrandomに置換して性能がどれくらい変わるのか
'''
import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
from sas import handling_data, util
from analysis.util import *
import evaluator_cls
from torch.utils import data
import torch
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--include_zero_score", "-izs", dest="include_zero_score", default=False,
    #                     action='store_true', help="By using this, zero score answer is taken into account in the justification identification")
    parser.add_argument("--train_info_path", "-info")
    parser.add_argument("--print-debug-info", "-debug",
                        dest="debug", default=False, action='store_true', help="")
    parser.add_argument("--attention_type", default='random',
                        choices=["random", "reversed"], help="attentionをどのタイプに置換するのか")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)
    info = pickle.load(open(args.train_info_path, "rb"))

    # modelのロード
    # model = load_model(info) if not args.ERASER else load_model_eraser(info)
    model = load_model_eraser(info)
    model = model.to(info.device)
    model.print_model_info()
    logger.info(f"Load pretrained model from {info.model_path}")

    # seedの固定
    seed = int(args.train_info_path.split('/')[-2])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # データのロード
    if info.BERT:
        from sas.input_data import input_data_ps_BERT as input_data
    else:
        from sas.input_data import input_data_ps as input_data

    (train_x, train_y, train_mask, train_attention, attention_flag), (dev_x, dev_y,
                                                                      dev_mask, dev_attention), (test_x, test_y, test_mask, test_attention) = input_data(info)

    from Dataset import part_scoring_set
    dataset = part_scoring_set(
        train_x, train_y, train_mask, train_attention, attention_flag)
    eval_dataloader = data.DataLoader(
        dataset, **{'batch_size': 32, 'shuffle': False, 'num_workers': 0})

    criterion = torch.nn.CrossEntropyLoss()
    item_num = util.get_item_num(info.item, info.main_factors)

    eval = evaluator_cls.EvaluatorForItem(
        info, dev_x, test_x, dev_y, test_y, dev_mask, test_mask, dev_attention, test_attention, item_num, criterion, None)

    model.eval()
    with torch.no_grad():
        (dev_qwks, test_qwks, dev_mses, test_mses, *_), train_hidden_states, train_golds, (dev_attentions, dev_hidden_states), (test_attentions,

                                                                                                                                test_hidden_states), d_pred, t_pred, dev_target, test_target, (dev_outputs, test_outputs), (_) = eval.evaluate(info, model, eval_dataloader, accuracy_mode=False, is_test=False, attention_type=args.attention_type)
    output_path = f"{info.out_dir}_{args.attention_type}_attn_evaluation_result.json"
    output = {"dev_qwk": dev_qwks, "dev_mse:": dev_mses,
              "test_qwk": test_qwks, "test_mse": test_mses}
    json.dump(output, open(output_path, 'w'))


if __name__ == '__main__':
    main()
