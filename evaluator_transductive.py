from sas.quadratic_weighted_kappa import linear_weighted_kappa as lwk
from sas.quadratic_weighted_kappa import quadratic_weighted_kappa as qwk
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import numpy as np
import math
import os
import sys
import torch
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

logger = logging.getLogger(__name__)


def squeeze_list(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)

    return flat_list


class Evaluator():
    def __init__(self, info, dev_x, test_x, dev_y, test_y, dev_m, test_m,dev_a,test_a):
        self.dev_x = dev_x
        self.test_x = test_x
        self.dev_y_org = dev_y
        self.info = info

        self.test_y_org = test_y
        self.dev_y = dev_y
        self.test_y = test_y
        self.dev_m = dev_m
        self.test_m = test_m
        self.dev_a = dev_a
        self.test_a =test_a
        self.high = info.high
        self.low = 0
        self.best_dev_qwks = {"Hol": -1.0}
        self.best_test_qwks = {"Hol": -1.0}
        self.test_attention_weight = []
        self.test_hidden_states = []
        self.dev_attention_weight = []
        self.dev_hidden_states = []
        self.neigh = KNeighborsClassifier(n_neighbors=1)

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
        try:
            dev_qwk = qwk(self.dev_y_org[factor_num],
                          dev_pred_int, self.low, self.high)
            test_qwk = qwk(self.test_y_org[factor_num],
                           test_pred_int, self.low, self.high)
            dev_lwk = lwk(self.dev_y_org[factor_num],
                          dev_pred_int, self.low, self.high)
            test_lwk = lwk(self.test_y_org[factor_num],
                           test_pred_int, self.low, self.high)
        except:
            dev_qwk=0
            test_qwk=0
            dev_lwk=0
            test_lwk=0
        return dev_qwk, test_qwk, dev_lwk, test_lwk

    def calc_mse(self, dev_pred, test_pred, factor_num,high):

        dev_pred_int = np.rint(dev_pred).astype('int32')
        test_pred_int = np.rint(test_pred).astype('int32')
        dev_mse = np.mean(np.square((self.dev_y_org[factor_num].numpy() -
                                    dev_pred_int) / high))
        test_mse = np.mean(np.square((self.test_y_org[factor_num].numpy() -
                                     test_pred_int) / high))
        dev_rmse = np.sqrt(dev_mse)
        test_rmse = np.sqrt(test_mse)
        return (dev_mse, dev_rmse), (test_mse, test_rmse)

    def print_info(self, qwks, mses, type, log=logger):

        msg = f"[{type}] QWK (RMSE) \t Hol {qwks['Hol']: .3f} ({mses['Hol'][1]:.3f}), "

        for i, factor in enumerate(self.info.main_factors):
            msg += f"{factor}: {qwks[factor]:.3f} ({mses[factor][1]:.3f}), "

        log.info(msg[:-2])

    def evaluate(self, info, model, train_data_loader, print_info=False, all=False, part=True):
        best = False
        dev_qwks = dict()
        test_qwks = dict()
        dev_mses = dict()
        test_mses = dict()

        # 指標の計算はcpuで処理
        model.eval()
        with torch.no_grad():

            dev_outputs = model(self.dev_x, token_type_ids=None,
                                attention_mask=self.dev_m,weight=self.dev_a)
            test_outputs = model(self.test_x, token_type_ids=None,
                                 attention_mask=self.test_m,weight=self.test_a)
            train_hidden_states = [torch.zeros(
                info.train_size, info.rnn_dim) for k in info.ps_labels]
            train_golds = [torch.zeros(
                info.train_size) for k in info.ps_labels]


            index = 0
            for num, (batch_x, batch_y, batch_m, batch_a) in enumerate(train_data_loader):
                outputs = model(batch_x.to(info.device),token_type_ids=None, attention_mask=batch_m.to(info.device))
                for i, label in enumerate(info.ps_labels):
                    train_hidden_states[i][index:index + batch_x.shape[0], :] = outputs[2][0][i]
                    train_golds[i][index:index + batch_x.shape[0]] = batch_y[i + 1]

                index += batch_x.shape[0]

        dev_pred = dev_outputs[0]
        dev_hidden_states = dev_outputs[2][0]
        test_pred = test_outputs[0]
        test_hidden_states = test_outputs[2][0]

        dev_dist = []
        test_dist = []
        dev_evidence = []
        test_evidence = []

        ds, ts = 0.0, 0.

        for i, factor in enumerate(info.main_factors):
            self.neigh.fit(train_hidden_states[i].cpu().detach(
            ).numpy(), train_golds[i].cpu().detach().numpy())

            d_dist, d_pred_id = self.neigh.kneighbors(
                dev_hidden_states[i].cpu().detach().numpy(), n_neighbors=3)
            # d_dist = d_dist[:,0]
            # d_pred_id = d_pred_id[:,0]
            d_pred = train_golds[i].cpu().detach().numpy()[d_pred_id[:,0]]
            t_dist, t_pred_id = self.neigh.kneighbors(
                test_hidden_states[i].cpu().detach().numpy(), n_neighbors=3)
            # t_dist = t_dist[:,0]
            # t_pred_id = t_pred_id[:,0]
            t_pred = train_golds[i].cpu().detach().numpy()[t_pred_id[:,0]]
            dev_dist.append(d_dist)
            test_dist.append(t_dist)
            dev_evidence.append(d_pred_id)
            test_evidence.append(t_pred_id)

            # print(d_pred)
            dev_pred[i] = d_pred
            test_pred[i] = t_pred

            # d_pred = np.argmax(
            #    dev_pred[i].softmax(1).cpu().detach().numpy(), axis=1)
            # t_pred = np.argmax(
            #    test_pred[i].softmax(1).cpu().detach().numpy(), axis=1)

            # logger.debug(dev_pred[i].softmax(1).cpu().detach().numpy().max(1))

            dev_qwks[factor], test_qwks[factor], dev_lwk, test_lwk = self.calc_qwk(
                d_pred, t_pred, self.factor2num(factor))

            dev_mses[factor], test_mses[factor] = self.calc_mse(
                d_pred, t_pred, self.factor2num(factor),self.info.ps_labels[i][1])

            # dev_qwks[factor] = d
            # test_qwks[factor] = t
            ds += d_pred
            ts += t_pred

        self.dev_pred = ds
        self.test_pred = ts
        for i in range(len(info.ded_factors)):
            num = len(info.main_factors) + i

            self.neigh.fit(train_hidden_states[num].cpu().detach().numpy(), train_golds[num].cpu().detach().numpy())

            d_dist, d_pred_id = self.neigh.kneighbors(
                dev_hidden_states[num].cpu().detach().numpy(), n_neighbors=1)
            d_dist = squeeze_list(d_dist)
            d_pred_id = squeeze_list(d_pred_id)
            d_pred = train_golds[num].cpu().detach().numpy()[d_pred_id]
            t_dist, t_pred_id = self.neigh.kneighbors(
                test_hidden_states[num].cpu().detach().numpy(), n_neighbors=1)
            t_dist = squeeze_list(t_dist)
            t_pred_id = squeeze_list(t_pred_id)
            t_pred = train_golds[num].cpu().detach().numpy()[t_pred_id]

            dev_pred[num] = d_pred
            test_pred[num] = t_pred
            dev_dist.append(d_dist)
            test_dist.append(t_dist)
            dev_evidence.append(d_pred_id)
            test_evidence.append(t_pred_id)
            ds += d_pred
            ts += t_pred

        np.where(ds >= 0, ds, 0)
        np.where(ts >= 0, ts, 0)

        dev_qwks["Hol"], test_qwks["Hol"], dev_lwk, test_lwk = self.calc_qwk(
            ds, ts, 0)
        dev_mses["Hol"], test_mses["Hol"] = self.calc_mse(
            ds, ts, 0,self.info.high)

        '''
        self.dev_prs, self.test_prs, self.dev_spr, self.test_spr, self.dev_tau, self.test_tau = self.calc_correl(
            self.ds, self.ts)
        '''
        self.print_info(dev_qwks, dev_mses, type="Dev")
        self.print_info(test_qwks, test_mses, type="Test")

        return (dev_qwks, test_qwks, dev_mses, test_mses), dev_outputs[1:], test_outputs[1:], dev_pred, test_pred, self.dev_y, self.test_y, (dev_dist, test_dist), (dev_evidence, test_evidence)
