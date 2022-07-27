from sas.quadratic_weighted_kappa import linear_weighted_kappa as lwk
from sas.quadratic_weighted_kappa import quadratic_weighted_kappa as qwk
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import numpy as np
import math
import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

logger = logging.getLogger(__name__)


class Evaluator():
    def __init__(self, info, dev_x, test_x, dev_y, test_y, dev_m, test_m):
        self.dev_x = dev_x
        self.test_x = test_x
        self.dev_y_org = dev_y
        self.info = info
        self.train_x = train_x


        self.test_y_org = test_y
        self.dev_y = dev_y
        self.test_y = test_y
        self.dev_m = dev_m
        self.test_m = test_m
        self.high = info.high
        self.low = 0
        self.best_dev_qwks = {"Hol": -1.0}
        self.best_test_qwks = {"Hol": -1.0}
        self.test_attention_weight = []
        self.test_hidden_states = []
        self.dev_attention_weight = []
        self.dev_hidden_states = []

    def calc_correl(self, dev_pred, test_pred):
        dev_prs, _ = pearsonr(dev_pred, self.dev_y_org)
        test_prs, _ = pearsonr(test_pred, self.test_y_org)
        dev_spr, _ = spearmanr(dev_pred, self.dev_y_org)
        test_spr, _ = spearmanr(test_pred, self.test_y_org)
        dev_tau, _ = kendalltau(dev_pred, self.dev_y_org)
        test_tau, _ = kendalltau(test_pred, self.test_y_org)
        return dev_prs, test_prs, dev_spr, test_spr, dev_tau, test_tau

    def factor2num(self, factor):
        assert ord('A') <= ord(factor) and ord('Z') >= ord(factor)
        return ord(factor) - ord('A') + 1

    def calc_qwk(self, dev_pred, test_pred, factor_num):
        # Kappa only supports integer values

        dev_pred_int = np.rint(dev_pred).astype('int32')
        test_pred_int = np.rint(test_pred).astype('int32')
        dev_qwk = qwk(self.dev_y_org[factor_num],
                      dev_pred_int, self.low, self.high)
        test_qwk = qwk(self.test_y_org[factor_num],
                       test_pred_int, self.low, self.high)
        dev_lwk = lwk(self.dev_y_org[factor_num],
                      dev_pred_int, self.low, self.high)
        test_lwk = lwk(self.test_y_org[factor_num],
                       test_pred_int, self.low, self.high)
        return dev_qwk, test_qwk, dev_lwk, test_lwk

    def calc_mse(self, dev_pred, test_pred, factor_num):
        # バグが有る

        dev_pred_int = np.rint(dev_pred).astype('int32')
        test_pred_int = np.rint(test_pred).astype('int32')
        dev_mse = np.mean(np.square(self.dev_y_org[factor_num].numpy() / self.high -
                                    dev_pred_int / self.high))
        test_mse = np.mean(np.square(self.test_y_org[factor_num].numpy() / self.high -
                                     test_pred_int / self.high))
        dev_rmse = np.sqrt(dev_mse)
        test_rmse = np.sqrt(test_mse)
        return (dev_mse, dev_rmse), (test_mse, test_rmse)

    def print_info(self, qwks, mses, type, log=logger):

        msg = f"[{type}] QWK (RMSE) \t Hol {qwks['Hol']: .3f} ({mses['Hol'][1]:.3f}), "

        for i, factor in enumerate(self.info.main_factors):
            msg += f"{factor}: {qwks[factor]:.3f} ({mses[factor][1]:.3f}), "

        log.info(msg[:-2])

    def evaluate(self, info, model, print_info=False, all=False, part=True):
        best = False
        dev_qwks = dict()
        test_qwks = dict()
        dev_mses = dict()
        test_mses = dict()

        # 指標の計算はcpuで処理
        model.eval()
        with torch.no_grad():
            dev_outputs = model(self.dev_x, self.dev_y_org, token_type_ids=None,
                                attention_mask=self.dev_m)
            test_outputs = model(self.test_x, self.test_y_org, token_type_ids=None,
                                 attention_mask=self.test_m)

        dev_pred = dev_outputs[0]
        test_pred = test_outputs[0]

        ds, ts = 0.0, 0.

        for i, factor in enumerate(info.main_factors):

            d_pred = np.argmax(
                dev_pred[i].softmax(1).cpu().detach().numpy(), axis=1)
            t_pred = np.argmax(
                test_pred[i].softmax(1).cpu().detach().numpy(), axis=1)
            # logger.debug(dev_pred[i].softmax(1).cpu().detach().numpy().max(1))

            dev_qwks[factor], test_qwks[factor], dev_lwk, test_lwk = self.calc_qwk(
                d_pred, t_pred, self.factor2num(factor))

            dev_mses[factor], test_mses[factor] = self.calc_mse(
                d_pred, t_pred, self.factor2num(factor))

            #dev_qwks[factor] = d
            #test_qwks[factor] = t

            ds += d_pred
            ts += t_pred

        self.dev_pred = ds
        self.test_pred = ts
        for i in range(len(info.ded_factors)):
            num = len(info.main_factors) + i

            d_pred = np.argmax(dev_pred[num].softmax(
                1).cpu().detach().numpy(), axis=1)
            t_pred = np.argmax(test_pred[num].softmax(
                1).cpu().detach().numpy(), axis=1)
            ds -= d_pred
            ts -= t_pred

            dev_qwks["Hol"], test_qwks["Hol"], dev_lwk, test_lwk = self.calc_qwk(
                ds, ts, 0)
            dev_mses["Hol"], test_mses["Hol"] = self.calc_mse(
                ds, ts, 0)
        '''
        self.dev_prs, self.test_prs, self.dev_spr, self.test_spr, self.dev_tau, self.test_tau = self.calc_correl(
            self.ds, self.ts)
        '''
        self.print_info(dev_qwks, dev_mses, type="Dev")
        self.print_info(test_qwks, test_mses, type="Test")

        return (dev_qwks, test_qwks, dev_mses, test_mses), dev_outputs[1:], test_outputs[1:], dev_pred, test_pred, self.dev_y, self.test_y
