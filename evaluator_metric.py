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
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

logger = logging.getLogger(__name__)


def squeeze_list(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)

    return flat_list


class Evaluator():
    def __init__(self, info, dev_x, test_x, dev_y, test_y, dev_m, test_m):
        self.dev_x = dev_x
        self.test_x = test_x
        self.dev_y_org = dev_y
        self.info = info

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
        dev_qwk = qwk(self.dev_y_org[factor_num],
                      dev_pred_int, self.low, self.high)
        test_qwk = qwk(self.test_y_org[factor_num],
                       test_pred_int, self.low, self.high)
        dev_lwk = lwk(self.dev_y_org[factor_num],
                      dev_pred_int, self.low, self.high)
        test_lwk = lwk(self.test_y_org[factor_num],
                       test_pred_int, self.low, self.high)
        return dev_qwk, test_qwk, dev_lwk, test_lwk

    def calc_mse(self, dev_pred, test_pred, factor_num, high):
        # バグが有る

        dev_pred_int = np.rint(dev_pred).astype('int32')
        test_pred_int = np.rint(test_pred).astype('int32')
        squared_dev = np.mean(np.square((self.dev_y_org[factor_num].numpy() -
                                         dev_pred_int) / high))
        squared_test = np.mean(np.square((self.test_y_org[factor_num].numpy() -
                                          test_pred_int) / high))
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
                                attention_mask=self.dev_m)
            test_outputs = model(self.test_x, token_type_ids=None,
                                 attention_mask=self.test_m)
            train_hidden_states = [torch.zeros(
                info.train_size, info.rnn_dim) for k in info.ps_labels]
            train_golds = [torch.zeros(
                info.train_size) for k in info.ps_labels]

            index = 0

            for num, (batch_x, batch_y, batch_m, batch_a) in enumerate(train_data_loader):
                outputs = model(batch_x.to(
                    info.device), token_type_ids=None, attention_mask=batch_m.to(info.device))
                for i, label in enumerate(info.ps_labels):
                    train_hidden_states[i][index:index +
                                           batch_x.shape[0], :] = outputs[2][0][i]
                    train_golds[i][index:index +
                                   batch_x.shape[0]] = batch_y[i + 1]

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
                dev_hidden_states[i].cpu().detach().numpy(), n_neighbors=1)
            d_dist = squeeze_list(d_dist)
            d_pred_id = squeeze_list(d_pred_id)
            d_pred = train_golds[i].cpu().detach().numpy()[d_pred_id]

            t_dist, t_pred_id = self.neigh.kneighbors(
                test_hidden_states[i].cpu().detach().numpy(), n_neighbors=1)
            t_dist = squeeze_list(t_dist)
            t_pred_id = squeeze_list(t_pred_id)
            t_pred = train_golds[i].cpu().detach().numpy()[t_pred_id]
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
                d_pred, t_pred, self.factor2num(factor), self.info.ps_labels[i][1])

            # dev_qwks[factor] = d
            # test_qwks[factor] = t
            ds += d_pred
            ts += t_pred

        self.dev_pred = ds
        self.test_pred = ts
        for i in range(len(info.ded_factors)):
            num = len(info.main_factors) + i

            self.neigh.fit(train_hidden_states[num].cpu().detach(
            ).numpy(), train_golds[num].cpu().detach().numpy())

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
            ds, ts, 0, self.info.high)

        '''
        self.dev_prs, self.test_prs, self.dev_spr, self.test_spr, self.dev_tau, self.test_tau = self.calc_correl(
            self.ds, self.ts)
        '''
        self.print_info(dev_qwks, dev_mses, type="Dev")
        self.print_info(test_qwks, test_mses, type="Test")

        return (dev_qwks, test_qwks, dev_mses, test_mses), dev_outputs[1:], test_outputs[1:], dev_pred, test_pred, self.dev_y, self.test_y, (dev_dist, test_dist), (dev_evidence, test_evidence)


class EvaluatorForItemScoring():
    def __init__(self, info, dev_x, test_x, dev_y, test_y, dev_m, test_m, dev_a, test_a, item_num, affin, criterion, normalize=False, ):
        self.dev_x = dev_x
        self.test_x = test_x
        self.dev_y_org = dev_y
        self.info = info
        self.item_num = item_num

        self.test_y_org = test_y
        self.dev_y = dev_y
        self.test_y = test_y
        self.dev_m = dev_m
        self.test_m = test_m
        self.dev_a = dev_a
        self.test_a = test_a
        self.high = info.high
        self.low = 0
        self.best_dev_qwks = {"Hol": -1.0}
        self.best_test_qwks = {"Hol": -1.0}
        self.test_attention_weight = []
        self.test_hidden_states = []
        self.dev_attention_weight = []
        self.dev_hidden_states = []
        self.neigh = KNeighborsClassifier(n_neighbors=1)
        self.affin = affin
        self.normalize = normalize
        self.criterion = criterion

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

    def calc_mse(self, dev_pred, test_pred, factor_num, high):

        dev_pred_int = np.rint(dev_pred).astype('int32')
        test_pred_int = np.rint(test_pred).astype('int32')
        squared_dev = np.mean(np.square((self.dev_y_org[factor_num].numpy() -
                                         dev_pred_int) / high))
        squared_test = np.mean(np.square((self.test_y_org[factor_num].numpy() -
                                          test_pred_int) / high))
        dev_mse = np.mean(squared_dev)
        test_mse = np.mean(squared_test)
        dev_rmse = np.mean(np.sqrt(squared_dev))
        test_rmse = np.mean(np.sqrt(squared_test))
        return (dev_mse, dev_rmse), (test_mse, test_rmse)

    def print_info(self, qwks, mses, item_num, type, log=logger):

        msg = f"[{type}] QWK (RMSE) \t Item {self.info.main_factors[item_num]}: {qwks:.3f} ({mses[1]:.3f})"

        log.info(msg)

    def evaluate(self, info, model, train_data_loader, print_info=False, all=False, part=True):
        best = False
        dev_qwks = 0
        test_qwks = 0
        dev_mses = 0
        test_mses = 0

        # 指標の計算はcpuで処理
        model.eval()
        with torch.no_grad():
            if self.info.oracle:
                # justificationを与えて推論
                dev_outputs = model(self.dev_x, token_type_ids=None,
                                    attention_mask=self.dev_m, justification=self.dev_a)
                test_outputs = model(self.test_x, token_type_ids=None,
                                     attention_mask=self.test_m, justification=self.test_a)
            else:
                dev_outputs = model(
                    self.dev_x, token_type_ids=None, attention_mask=self.dev_m)
                test_outputs = model(
                    self.test_x, token_type_ids=None, attention_mask=self.test_m)
            # 勾配計算のためにgradを保存)
            # dev_y_org = self.dev_y_org[self.item_num + 1].to(info.device)
            # test_y_org = self.test_y_org[self.item_num + 1].to(info.device)

            # dev_output = model.get_embeddings(
            #     self.dev_x, token_type_ids=None, attention_mask=self.dev_m)
            # dev_outputs = model.get_grads(
            #     output=dev_output, attention_mask=self.dev_m, labels=dev_y_org)
            # test_output = model.get_embeddings(
            #     self.test_x, token_type_ids=None, attention_mask=self.test_m)
            # test_outputs = model.get_grads(
            #     output=test_output, attention_mask=self.test_m, labels=test_y_org)

            # 勾配を求める
            # dev_loss = self.criterion(
            #     dev_outputs[0], dev_y_org, info.ps_labels[self.item_num][1])  # self.item+1する必要はない
            # test_loss = self.criterion(
            #     test_outputs[0], test_y_org, info.ps_labels[self.item_num][1])
            # if dev_loss[1] != 0:
            #     dev_loss[0].backward()
            #     dev_grads = dev_output.grad
            # else:
            #     dev_grads = torch.zeros_like(dev_output)
            # if test_loss[1] != 0:
            #     test_loss[0].backward()
            #     test_grads = test_output.grad
            # else:
            #     test_grads = torch.zeros_like(test_output)
            dev_grads = None
            test_grads = None

            # 推論用の中間表現保存用
            # train_hidden_states = torch.zeros(
            #     info.train_size, self.info.rnn_dim)
            train_hidden_states = torch.zeros(
                info.train_size, info.outdim)
            train_golds = torch.zeros(info.train_size)

            index = 0

            for num, (batch_x, batch_y, batch_m, batch_a, batch_f) in enumerate(train_data_loader):
                outputs = model(batch_x.to(
                    info.device), token_type_ids=None, attention_mask=batch_m.to(info.device))

                i = self.item_num
                if self.affin:
                    train_hidden_states[index:index +
                                        batch_x.shape[0], :] = outputs[0]
                    train_golds[index:index +
                                batch_x.shape[0]] = batch_y[i + 1]
                    dev_hidden_states = dev_outputs[0]
                    test_hidden_states = test_outputs[0]
                else:
                    train_hidden_states[index:index +
                                        batch_x.shape[0], :] = outputs[2][0]
                    train_golds[index:index +
                                batch_x.shape[0]] = batch_y[i + 1]
                    dev_hidden_states = dev_outputs[2][0]
                    test_hidden_states = test_outputs[2][0]
                train_attentions = outputs[1]
                index += batch_x.shape[0]

        ds, ts = 0.0, 0.

        i = self.item_num

        inference = 'knn'
        if inference == "knn":
            self.neigh.fit(self.normalize_vector(np.nan_to_num(train_hidden_states.cpu().detach().numpy())),
                           train_golds.cpu().detach().numpy())

            d_dist, d_pred_id = self.neigh.kneighbors(
                self.normalize_vector(np.nan_to_num(dev_hidden_states.cpu().detach().numpy())), n_neighbors=1)
            d_dist = squeeze_list(d_dist)
            d_pred_id = squeeze_list(d_pred_id)
            d_pred = train_golds.cpu().detach().numpy()[d_pred_id]
            t_dist, t_pred_id = self.neigh.kneighbors(
                self.normalize_vector(np.nan_to_num(test_hidden_states.cpu().detach().numpy())), n_neighbors=1)
            t_dist = squeeze_list(t_dist)
            t_pred_id = squeeze_list(t_pred_id)
            t_pred = train_golds.cpu().detach().numpy()[t_pred_id]
            dev_dist = d_dist
            test_dist = t_dist
            dev_evidence = d_pred_id
            test_evidence = t_pred_id

            dev_pred = d_pred
            test_pred = t_pred
        elif inference == "NED":
            # 根拠事例は出せないのでどうするか

            # devで全答案との距離を計算

            train_hidden_states = train_hidden_states.to(info.device)
            train_golds = train_golds.to(info.device)
            dev_hidden_states = dev_hidden_states.to(info.device)
            test_hidden_states = test_hidden_states.to(info.device)

            max_score = int(torch.max(train_golds).item())

            exp_dist_matrix = torch.exp(-1 * torch.cdist(
                dev_hidden_states, train_hidden_states, p=2))
            sum_dist = torch.sum(exp_dist_matrix, dim=-1)
            d_pred = torch.zeros(dev_hidden_states.size(0),
                                 max_score + 1)
            for score in range(max_score + 1):
                d_pred[:, score] = torch.sum(
                    exp_dist_matrix[:, train_golds == score], dim=-1)
            d_pred = torch.argmax(d_pred, dim=-1)

            exp_dist_matrix = torch.exp(-1 * torch.cdist(
                test_hidden_states, train_hidden_states, p=2))
            sum_dist = torch.sum(exp_dist_matrix, dim=-1)
            t_pred = torch.zeros(test_hidden_states.size(0),
                                 max_score + 1)
            for score in range(max_score + 1):
                t_pred[:, score] = torch.sum(
                    exp_dist_matrix[:, train_golds == score], dim=-1)
            t_pred = torch.argmax(t_pred, dim=-1)

            d_pred = d_pred.cpu().detach().numpy()
            t_pred = t_pred.cpu().detach().numpy()
            dev_pred = d_pred
            test_pred = t_pred

            dev_dist = None
            test_dist = None
            dev_evidence = None
            test_evidence = None

        dev_qwks, test_qwks, dev_lwk, test_lwk = self.calc_qwk(
            d_pred, t_pred, self.item_num + 1)

        dev_mses, test_mses = self.calc_mse(
            d_pred, t_pred, self.item_num + 1, self.info.ps_labels[i][1])

        ds += d_pred
        ts += t_pred

        self.dev_pred = ds
        self.test_pred = ts

        self.print_info(dev_qwks, dev_mses, self.item_num, type="Dev")
        self.print_info(test_qwks, test_mses, self.item_num, type="Test")

        return (dev_qwks, test_qwks, dev_mses, test_mses), (train_hidden_states, train_golds, train_attentions), (dev_outputs, test_outputs), (dev_pred, test_pred), (self.dev_y, self.test_y), (dev_dist, test_dist), (dev_evidence, test_evidence), (dev_grads, test_grads)

    def normalize_vector(self, vector):
        if self.normalize:
            vector = normalize(vector, norm='l2', axis=1)
        return vector
