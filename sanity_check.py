from itertools import islice
import time
from analysis.util import *
from interpretability.explanation import Explanation
import json
import pickle
from typing import List
from sas import util
import argparse
import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import logzero
from logzero import logger
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SanityCheck:
    def __init__(self, model, device, mask_id, batch_size=64, debug=False, include_zero_score=False):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.debug = debug
        self.mask_id = mask_id
        self.batch_size = batch_size
        self.include_zero_score = include_zero_score

    def _predict(self, input_tensor, mask):
        _input_tensor = input_tensor.detach().clone().to(self.device)
        _mask = mask.detach().clone().to(self.device)
        self.model.eval()
        with torch.no_grad():
            # raise ValueError(
            #     "you haven't inmplement pred func when instance based yet")
            output = self.model.predict(
                ids=_input_tensor, attention_mask=_mask)

        orig_prob = torch.nn.functional.softmax(output, dim=-1).squeeze(0)
        orig_label = torch.argmax(output, dim=-1)
        return orig_label, orig_prob

    @staticmethod
    def _remove(input_tensor, remove_idx, padding_length, destroy=True):
        if destroy:
            _input_tensor = input_tensor
        else:
            _input_tensor = input_tensor.detach().clone()
        _input_tensor[remove_idx] = -1
        removed = _input_tensor[_input_tensor >= 0.0]
        padding_tensor = torch.zeros(
            padding_length, device="cpu", dtype=torch.long)
        return torch.cat((removed, padding_tensor))

    def _mask(self, input_tensor, remove_idx, destroy=True):
        if destroy:
            _input_tensor = input_tensor
        else:
            _input_tensor = input_tensor.detach().clone()
        _input_tensor[remove_idx] = self.mask_id
        return _input_tensor.detach().clone()

    def _erase_important_features(self, input_tensor, _map, mask=False, most=True):
        _input_tensor = input_tensor.detach().clone().squeeze(0)
        tensor_size = _input_tensor.shape[0]

        # sort and sum
        sorted_args = torch.argsort(_map, descending=most)

        # make removed tensor
        new_tensor = torch.zeros((tensor_size, tensor_size), dtype=torch.long)
        for idx, remove_idx in enumerate(sorted_args):
            # remove
            if mask:
                new_tensor[idx] = self._mask(
                    _input_tensor, remove_idx, destroy=True)
            else:
                new_tensor[idx] = self._remove(
                    _input_tensor, remove_idx, idx + 1, destroy=True)

        return new_tensor, sorted_args

    def _aopc_score(self, eraser_prob, orig_prob, orig_label):
        scores = ((eraser_prob - orig_prob) * (-1)
                  ).T[orig_label].view(-1).detach().cpu().numpy()
        return scores

    def _flip_check(self, eraser_labels, orig_label, removed_size):
        if not self.include_zero_score:
            if orig_label == 0:
                return np.NaN, np.NaN
        for i, label in enumerate(eraser_labels):
            if orig_label != label:
                _ratio = (i + 1) / removed_size
                if _ratio > 1.0:
                    # _ratio = -1.0
                    # なぜ-1にしているのか？
                    _ratio = 1.0
                return _ratio, i + 1
        # なぜ-1にしているのか？
        return 1.0, removed_size

    def _check(self, erased_tensor, mask):
        self.model.eval()
        with torch.no_grad():
            eraser_output = None
            for batch_index in range(0, len(erased_tensor), self.batch_size):
                b_e_t = erased_tensor[batch_index:batch_index +
                                      self.batch_size].to(self.device)
                b_m = mask[batch_index:batch_index +
                           self.batch_size].to(self.device)
                if len(b_m) == 1:
                    b_e_t = b_e_t.unsqueeze(0)
                    b_m = b_m.unsqueeze(0)
                if eraser_output is None:
                    # raise ValueError(
                    #     "you haven't inmplement pred func when instance based yet")
                    eraser_output = self.model.predict(
                        ids=b_e_t, attention_mask=b_m)
                else:
                    # raise ValueError(
                    #     "you haven't inmplement pred func when instance based yet")
                    eraser_output = torch.cat(
                        (eraser_output, self.model.predict(ids=b_e_t, attention_mask=b_m)), dim=0)
        return eraser_output

    def _calc_aopc_and_flip_score(self, _text_tensor, _mask, length, _map, most=True):
        input_tensor = _text_tensor.unsqueeze(0).detach().clone()
        mask = _mask.unsqueeze(0).detach().clone()

        # predict
        orig_label, orig_prob = self._predict(input_tensor, mask)

        # mask
        erased_tensor, sorted_args = self._erase_important_features(
            input_tensor, _map, mask=True, most=most)
        erased_mask = mask.repeat((len(erased_tensor), 1))
        eraser_output = self._check(erased_tensor, erased_mask)
        eraser_labels = torch.argmax(eraser_output, dim=1)
        eraser_prob = torch.nn.functional.softmax(eraser_output, dim=1)

        # calculate score
        aopc_scores = self._aopc_score(
            eraser_prob, orig_prob, orig_label).tolist()
        flip_scores, flip_time = self._flip_check(
            eraser_labels, orig_label, length)

        # 何もmaskしない時のデータが0番地に入る
        return aopc_scores, flip_scores, [None] + sorted_args.cpu().numpy().tolist(), [orig_prob.to("cpu").numpy().tolist()] + eraser_prob.cpu().numpy().tolist(), flip_time

    def _calc_faithfulness(self, _source, _map):
        source = _source.unsqueeze(0).detach().clone()

        orig_label, orig_prob = self._predict(source)

        # mask
        repeat_source = source.repeat(source.shape[1], 1).to(self.device)
        for idx in range(repeat_source.shape[0]):
            repeat_source[idx][idx] = 0
            # 対角のidxを0にマスク

        output = self._check(repeat_source)
        prob = torch.nn.functional.softmax(output, dim=1)
        diff = self._aopc_score(prob, orig_prob, orig_label)

        f_score, p_value = stats.pearsonr(diff.cpu(), _map.cpu())
        return f_score, p_value

    def _sanity_check_for_aopc_and_flip(self, sources, targets, masks, lengths, maps):
        results_most, results_least = [], []
        for idx, (source, target, mask, length) in tqdm(enumerate(zip(sources, targets, masks, lengths))):
            _map = torch.tensor(maps[idx])
            most = self._calc_aopc_and_flip_score(
                source, mask, length, _map, most=True)
            least = self._calc_aopc_and_flip_score(
                source, mask, length, _map, most=False)

            # append
            results_most.append(most)
            results_least.append(least)

        return results_most, results_least

    def _sanity_check_for_faithfulness(self, sources, targets, mask, maps):
        f_scores, p_values = [], []
        # データ1個ずつ計算
        for idx, (source, target, _map) in tqdm(enumerate(zip(sources, targets, maps))):
            f_score, p_value = self._calc_faithfulness(source, _map)
            f_scores.append(f_score)
            p_values.append(p_value)
        return f_scores, p_values

    def _check_model_output_size(self, sources, targets, masks):
        size = []
        for idx, (source, target, mask) in tqdm(enumerate(zip(sources, targets, masks))):
            output, *_ = self.model(ids=source.unsqueeze(0).to(self.device),
                                    attention_mask=mask.unsqueeze(0).to(self.device))
            size.append(torch.norm(output).detach().item())
        return size

    def __call__(self, sources, targets, mask, explanations, limit=None):

        # debug & limit mode
        if self.debug:
            sources, targets, mask = sources[:10], targets[:10], mask[:10]
        elif limit is not None:
            sources, targets, mask = sources[:limit], targets[:limit], mask[:limit]

        results_df = pd.DataFrame()

        for name in explanations.columns:
            maps = explanations[name]

            if self.debug:
                maps = maps[:10]
            elif limit is not None:
                maps = maps[:limit]

            lengths = mask.sum(-1).tolist()

            results_most, results_least = self._sanity_check_for_aopc_and_flip(
                sources, targets, mask, lengths, maps)
            # f_scores, p_values = self._sanity_check_for_faithfulness(sources, targets, mask,  maps)
            size = self._check_model_output_size(sources, targets, mask)

            aopc_most, flip_most, args_most, flip_most_probs, flip_most_time = zip(
                *results_most)
            aopc_least, flip_least, args_least, flip_least_probs, flip_least_time = zip(
                *results_least)
            indice = [i for i in range(len(sources))]

            results = {"idx": indice}
            if not flip_most_probs:
                print(flip_most_probs)
                raise ValueError
            results.update(
                {"aopc_most": aopc_most, "flip_most": flip_most, "args_most": args_most, "flip_most_probs": flip_most_probs, "flip_most_time": flip_most_time, "length": lengths})
            results.update(
                {"aopc_least": aopc_least, "flip_least": flip_least, "args_least": args_least, "flip_least_probs": flip_least_probs, "flip_least_time": flip_least_time, "length": lengths})
            # results.update({"faithfulness": f_scores, "f_p": p_values})
            results.update({"output_size": size})

            df = pd.DataFrame(results)
            df["name"] = name

            results_df = pd.concat([results_df, df], ignore_index=False)

        return results_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_zero_score", "-izs", dest="include_zero_score", default=False,
                        action='store_true', help="By using this, zero score answer is taken into account in the justification identification")
    parser.add_argument("--train_info_path", "-info")
    parser.add_argument("--print-debug-info", "-debug",
                        dest="debug", default=False, action='store_true', help="")
    parser.add_argument("-ERASER", action='store_true',
                        default=False, help='use eraser dataset')
    parser.add_argument("--evaluate_size", '-es', type=int,
                        default=1000000000000, help='test data size to evaluate')
    parser.add_argument("--mask_type", '-mt', choices=['zero', 'rand'], default="zero",
                        help="how to mask. choose [zero, rand]. zero replace token 0 vector. rand replace token initalized embedding")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    info = pickle.load(open(args.train_info_path, "rb"))
    np.random.seed(info.seed)

    # explanationデータをロード
    test_explanation_df = get_explanation_pickle_data(info, "test")

    # 評価データのサイズを指定
    eval_size = min(args.evaluate_size, len(test_explanation_df))

    test_explanation_df = test_explanation_df[:eval_size]

    # 訓練データと予測データのjsonファイルを取得
    _, test_gold_justification_list, _, test_pred_score_list, _ = get_sas_list(
        info, "test", size=eval_size) if not args.ERASER else get_eraser_list(info, 'test', size=eval_size)

    # modelのロード
    # model = load_model(info) if not args.ERASER else load_model_eraser(info)
    model = load_model_eraser(info)
    model = model.to(device)
    model.print_model_info()

    # データのロード
    (_), (_), (test_x, test_y, test_mask, _) = load_data(info, eval_size)

    # maskの種類を選択
    if info.BERT:
        mask_id = info.vocab["[MASK]"]
    else:
        if args.mask_type == "zero":
            mask_id = info.vocab["<zero>"]
        elif args.mask_type == "rand":
            mask_id = info.vocab["<rand>"]
        else:
            raise ValueError

    sc = SanityCheck(model, device, mask_id, debug=args.debug)
    results = sc(test_x, test_y, test_mask, test_explanation_df, limit=None)

    if not args.debug:
        pickle.dump(results, open(info.out_dir + "_test" +
                                  "_sanity_check.pickle", "wb"))
    else:
        pickle.dump(results, open(info.out_dir + "_test" +
                                  "_sanity_check_debug.pickle", "wb"))


if __name__ == "__main__":
    main()
