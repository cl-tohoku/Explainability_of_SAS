'''
解釈法での特徴マップを作成
'''

import numpy as np
import argparse
from util import logger as L
from sas import util
from typing import List
import pickle
import json
# from getinfo import TrainInfo
from logging import getLogger
# import getinfo
# from Dataset import part_scoring_set
# from collections import defaultdict
from interpretability.explanation import Explanation
from analysis.util import *
logger = getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_info_path", "-info")
    parser.add_argument("--print-debug-info", "-debug",
                        dest="debug", default=False, action='store_true', help="")
    parser.add_argument("--instance_based", "-ib",
                        dest="ib", default=False, action='store_true', help="use instance based model")
    parser.add_argument("--evaluate_size", '-es', type=int,
                        default=1000000000000, help='test data size to evaluate')
    parser.add_argument("--use_gold", '-ug', default=False,
                        action='store_true', help="use gold label")
    args = parser.parse_args()
    return args


def make_explanation_file(info, explanation, x, y, mask, mode, prediction):
    if type(y) is not list:
        y = y.numpy().tolist()
    explanation_df = explanation(
        ids=x, attention_mask=mask, labels=y, prediction=prediction)
    pickle.dump(explanation_df, open(
        info.get_explanation_path(mode), "wb"))
    return explanation_df


def make_explanation_file_for_instancebase(info, explanation, x, y, mask, mode):
    # y = y.numpy().tolist()
    explanation_df = explanation.explanation_for_instacebase(
        ids=x, attention_mask=mask, labels=y, info=info)
    pickle.dump(explanation_df, open(
        info.get_explanation_path(mode), "wb"))
    return explanation_df


def main():
    args = parse_args()
    # pickleをロード
    info = pickle.load(open(args.train_info_path, "rb"))

    L.set_logger(
        out_dir=f"{info.out_dir}_test_interpretability", debug=args.debug)
    info.model_path = info.out_dir + "_best_checkpoint.pt"  # 暫定的に入れとく

    # モデルをロード
    model = load_model_eraser(info)
    logger.info(model)

    # データをロード
    _, (dev_x, dev_y, dev_mask, dev_attention), (test_x,
                                                 test_y, test_mask, test_attention) = load_data(info, eval_size=args.evaluate_size)

    if not args.use_gold:
        # 訓練データと予測データのjsonファイルを取得
        logger.info("use predictin label intsted of gold label")
        from analysis.util import get_sas_list
        _, _, _, test_y, _ = get_sas_list(
            info, "test", size=args.evaluate_size)
        _, _, _, dev_y, _ = get_sas_list(info, "dev", size=args.evaluate_size)
        # _, test_gold_justification_list, _, test_pred_score_list, _ = get_sas_list(info, "test", size=eval_size) if not args.ERASER else get_eraser_list(info, 'test',size=eval_size)
    else:
        logger.info("use gold label intsted of pred label")

    dataset = {"dev": (dev_x, dev_y, dev_mask),
               "test": (test_x, test_y, test_mask)}

    explanation = Explanation(model, info)

    for mode in ["dev", "test"]:
        make_explanation_file_for_instancebase(
            info, explanation, *dataset[mode], mode) if args.ib else make_explanation_file(info, explanation, *dataset[mode], mode, prediction=not args.use_gold)
    return


if __name__ == '__main__':
    main()
