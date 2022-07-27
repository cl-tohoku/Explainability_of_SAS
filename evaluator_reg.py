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

def convert_orignal_score(data_y,info):
    org_data = []
    org_data.append(data_y[0].astype('int32'))
    for i, factor in enumerate(info.factors):
        org_data.append(np.rint(data_y[i + 1] * info.ps_labels[i][1]).astype('int32'))
    return org_data


class Evaluator():
    def __init__(self, info, dev_x, test_x, dev_y, test_y, dev_m, test_m):
        self.dev_x = dev_x
        self.test_x = test_x
        self.info = info
        self.dev_y_org = convert_orignal_score(dev_y.cpu().numpy(),self.info)
        self.test_y_org = convert_orignal_score(test_y.cpu().numpy(),self.info)
        self.dev_y = dev_y.cpu().numpy()
        self.test_y = test_y.cpu().numpy()
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

        dev_qwk = qwk(self.dev_y_org[factor_num],dev_pred, self.low, self.high)
        test_qwk = qwk(self.test_y_org[factor_num],
                       test_pred, self.low, self.high)
        dev_lwk = lwk(self.dev_y_org[factor_num],
                      dev_pred, self.low, self.high)
        test_lwk = lwk(self.test_y_org[factor_num],
                       test_pred, self.low, self.high)
        return dev_qwk, test_qwk, dev_lwk, test_lwk

    def calc_mse(self, dev_pred, test_pred, factor_num):
        if factor_num == 0:
            high = self.high
        else:
            high = self.info.ps_labels[factor_num -1][1]
        squared_dev = np.square((self.dev_y[factor_num] - dev_pred) / high)
        squared_test = np.square((self.test_y[factor_num] - test_pred) /high)
        dev_mse = np.mean(squared_dev)
        test_mse = np.mean(squared_test)
        dev_rmse = np.mean(np.sqrt(squared_dev))
        test_rmse = np.mean(np.sqrt(squared_test))
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
            dev_outputs = model(self.dev_x, token_type_ids=None,
                                attention_mask=self.dev_m)
            test_outputs = model(self.test_x, token_type_ids=None,
                                 attention_mask=self.test_m)

        dev_pred = dev_outputs[0]
        test_pred = test_outputs[0]

        ds, ts = 0, 0

        for i, factor in enumerate(info.main_factors):

            d_pred = torch.squeeze(dev_pred[i]).cpu().detach().numpy()
            d_pred = np.rint(np.clip(d_pred,0,1) * info.ps_labels[i][1]).astype(np.int32)
            dev_pred[i] = d_pred
            t_pred = torch.squeeze(test_pred[i]).cpu().detach().numpy()
            t_pred = np.rint(np.clip(t_pred,0,1) * info.ps_labels[i][1]).astype(np.int32)
            test_pred[i] = t_pred
            # logger.debug(dev_pred[i].softmax(1).cpu().detach().numpy().max(1))
            dev_qwks[factor], test_qwks[factor], dev_lwk, test_lwk = self.calc_qwk(
                d_pred, t_pred, i+1)

            dev_mses[factor], test_mses[factor] = self.calc_mse(
                d_pred, t_pred, i+1)

            #dev_qwks[factor] = d
            #test_qwks[factor] = t

            ds += d_pred
            ts += t_pred

        self.dev_pred = ds
        self.test_pred = ts
        for i in range(len(info.ded_factors)):
            num = len(info.main_factors) + i

            d_pred = torch.squeeze(dev_pred[num]).cpu().detach().numpy()
            d_pred = np.rint(np.clip(d_pred,0,1) * info.ps_labels[num][1]).astype(np.int32)
            dev_pred[num] = d_pred
            t_pred = torch.squeeze(test_pred[num]).cpu().detach().numpy()
            t_pred = np.rint(np.clip(t_pred,0,1) * info.ps_labels[num][1]).astype(np.int32)
            test_pred[num] = t_pred
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



class EvaluatorForItem():
    def __init__(self, info, dev_x, test_x, dev_y, test_y, dev_m, test_m, item_num):
        self.dev_x = dev_x
        self.test_x = test_x
        self.info = info
        self.dev_y_org = convert_orignal_score(dev_y.cpu().numpy(),self.info)
        self.test_y_org = convert_orignal_score(test_y.cpu().numpy(),self.info)
        self.dev_y = dev_y.cpu().numpy()
        self.test_y = test_y.cpu().numpy()
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
        self.item_num = item_num

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


        dev_qwk = qwk(self.dev_y_org[factor_num],dev_pred, self.low, self.high)

        test_qwk = qwk(self.test_y_org[factor_num],test_pred, self.low, self.high)

        dev_lwk = lwk(self.dev_y_org[factor_num],
                      dev_pred, self.low, self.high)
        test_lwk = lwk(self.test_y_org[factor_num],
                       test_pred, self.low, self.high)

        return dev_qwk, test_qwk, dev_lwk, test_lwk

    def calc_mse(self, dev_pred, test_pred, factor_num):
        if factor_num == 0:
            high = self.high
        else:
            high = self.info.ps_labels[factor_num -1][1]
        squared_dev = np.square((self.dev_y[factor_num] - dev_pred) / self.info.ps_labels[factor_num - 1][1])
        squared_test = np.square((self.test_y[factor_num] - test_pred) /self.info.ps_labels[factor_num - 1][1])
        dev_mse = np.mean(squared_dev)
        test_mse = np.mean(squared_test)
        dev_rmse = np.mean(np.sqrt(squared_dev))
        test_rmse = np.mean(np.sqrt(squared_test))
        return (dev_mse, dev_rmse), (test_mse, test_rmse)

    def print_info(self, qwks, mses, item_num,type,log=logger):

        msg = f"[{type}] QWK (RMSE) \t Item {self.info.main_factors[item_num]}: {qwks:.3f} ({mses[1]:.3f})"

        log.info(msg)

    def evaluate(self, info, model, print_info=False, all=False, part=True):
        best = False
        dev_qwks = dict()
        test_qwks = dict()
        dev_mses = dict()
        test_mses = dict()

        # 指標の計算はcpuで処理
        model.eval()
        with torch.no_grad():
            dev_outputs = model(self.dev_x, token_type_ids=None,
                                attention_mask=self.dev_m)
            test_outputs = model(self.test_x, token_type_ids=None,
                                 attention_mask=self.test_m)

        dev_pred = dev_outputs[0]
        test_pred = test_outputs[0]

        ds, ts = 0.0, 0.

        i = self.item_num
        d_pred = torch.squeeze(dev_pred).cpu().detach().numpy()
        d_pred = np.rint(np.clip(d_pred,0,1) * info.ps_labels[i][1]).astype(np.int32)
        dev_pred = d_pred
        t_pred = torch.squeeze(test_pred).cpu().detach().numpy()
        t_pred = np.rint(np.clip(t_pred,0,1) * info.ps_labels[i][1]).astype(np.int32)
        test_pred = t_pred
        # logger.debug(dev_pred[i].softmax(1).cpu().detach().numpy().max(1))
        dev_qwks, test_qwks, dev_lwk, test_lwk = self.calc_qwk(d_pred, t_pred, i + 1)

        dev_mses, test_mses = self.calc_mse(d_pred, t_pred, i + 1)

        #dev_qwks[factor] = d
        #test_qwks[factor] = t

        ds += d_pred
        ts += t_pred

        self.dev_pred = ds
        self.test_pred = ts

        self.print_info(dev_qwks, dev_mses, self.item_num,type="Dev")
        self.print_info(test_qwks, test_mses, self.item_num,type="Test")
        return (dev_qwks, test_qwks, dev_mses, test_mses), dev_outputs[1:], test_outputs[1:], dev_pred, test_pred, self.dev_y, self.test_y
