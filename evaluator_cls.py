from sas.quadratic_weighted_kappa import linear_weighted_kappa as lwk
from sas.quadratic_weighted_kappa import quadratic_weighted_kappa as qwk
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import numpy as np
import math
import os
import sys
import torch
from typing import List
from loss import attn_loss

from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from train_item_scoring_model_eraser import regularization

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

logger = logging.getLogger(__name__)


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

        ds, ts = 0.0, 0.

        for i, factor in enumerate(info.main_factors):

            d_pred = np.argmax(dev_pred[i].softmax(
                1).cpu().detach().numpy(), axis=1)
            t_pred = np.argmax(test_pred[i].softmax(
                1).cpu().detach().numpy(), axis=1)
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

            d_pred = np.argmax(dev_pred[num].softmax(
                1).cpu().detach().numpy(), axis=1)
            t_pred = np.argmax(test_pred[num].softmax(
                1).cpu().detach().numpy(), axis=1)
            ds -= d_pred
            ts -= t_pred

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

        return (dev_qwks, test_qwks, dev_mses, test_mses), dev_outputs[1:], test_outputs[1:], dev_pred, test_pred, self.dev_y, self.test_y


class EvaluatorForItem():

    def __init__(self, info, dev_x, test_x, dev_y, test_y, dev_m, test_m, dev_attn, test_attn, item_num, criterion, method):
        from Dataset import part_scoring_set
        from torch.utils import data
        self.dev_x = dev_x
        self.test_x = test_x
        self.dev_y_org = dev_y
        self.info = info
        self.criterion = criterion
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
        self.item_num = item_num
        self.method = method

        dev_dataset = part_scoring_set(
            dev_x, dev_y, dev_m, dev_attn, torch.zeros_like(dev_y) == 1)
        self.dev_data_loader = data.DataLoader(
            dev_dataset, **{'batch_size': 32, 'shuffle': False, 'num_workers': 0})

        test_dataset = part_scoring_set(
            test_x, test_y, test_m, test_attn, torch.zeros_like(test_y) == 1)
        self.test_data_loader = data.DataLoader(
            test_dataset, **{'batch_size': 32, 'shuffle': False, 'num_workers': 0})

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

    def calc_accuracy(self, dev_pred, test_pred, factor_num):

        # 四捨五入
        dev_pred_int = np.rint(dev_pred).astype('int32')
        test_pred_int = np.rint(test_pred).astype('int32')
        # print(self.test_y_org[factor_num], dev_pred_int)
        dev_accuracy = accuracy_score(
            self.dev_y_org[factor_num], dev_pred_int)
        test_accuracy = accuracy_score(
            self.test_y_org[factor_num], test_pred_int)
        return dev_accuracy, test_accuracy

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

    def print_loss_info(self, score_loss, attn_loss, grad_loss, item_num, type, log=logger):

        msg = f"[{type}] LOSS \t Item {self.info.main_factors[item_num]}: score:{score_loss:.3f} attn:{attn_loss:.3f} grad:{grad_loss:.3f}"

        log.info(msg)

    # def  calc_justification_accuracy(gold_justications:List[List[bool]], pred_justications:List[List[bool]] ) -> List[float]:
    #     '''
    #     calculate accuracy between two justifications
    #     '''
    #     return

    # def calc_justification_f1(gold_justications:List[List[bool]], pred_justications:List[List[bool]] ) -> List[float]:
    #     '''
    #     calculate f1 between two justifications
    #     '''
    #     return

    # def calc_justification_f1(gold_justications:List[List[bool]], pred_justications:List[List[bool]] ) -> List[float]:
    #     accuracy_score

    def _evaluate(self, info, model, train_dataloader, mode, print_info=False, all=False, part=True, accuracy_mode=False):
        best = False
        qwks = dict()
        mses = dict()

        # 指標の計算はcpuで処理
        model.eval()
        with torch.no_grad():
            _pred = None
            data_loader = self.dev_data_loader if mode == "dev" else self.test_data_loader
            for batch_x, batch_y, batch_m, batch_a, batch_f in data_loader:
                batch_pred = model(batch_x.to(info.device), token_type_ids=None,
                                   attention_mask=batch_m.to(info.device), labels=batch_y[self.item_num + 1].to(info.device))[0]
                if _pred is None:
                    _pred = batch_pred
                else:
                    _pred = torch.cat((_pred, batch_pred))

        pred = np.argmax(_pred.softmax(1).cpu().detach().numpy(), axis=1)

        if accuracy_mode:
            if mode == 'dev':
                qwks, _ = self.calc_accuracy(
                    pred, pred, self.item_num + 1)
            else:
                _, qwks = self.calc_accuracy(
                    pred, pred, self.item_num + 1)
        else:
            if mode == "dev":
                qwks, _, lwk, _ = self.calc_qwk(
                    pred, pred, self.item_num + 1)
            else:
                _, qwks, _, lwk = self.calc_qwk(
                    pred, pred, self.item_num + 1)

        if mode == 'dev':
            mses, _ = self.calc_mse(
                pred, pred, self.item_num + 1, self.info.ps_labels[self.item_num][1])
            self.dev_pred = pred
            self.print_info(qwks, mses, self.item_num, type="Dev")
        else:
            _, mses = self.calc_mse(
                pred, pred, self.item_num + 1, self.info.ps_labels[self.item_num][1])
            self.test_pred = pred
            self.print_info(qwks, mses, self.item_num, type="Test")

        train_hidden_states = torch.zeros(info.train_size, self.info.outdim)
        train_golds = torch.zeros(info.train_size)
        index = 0

        # for num, (batch_x, batch_y, batch_m, batch_a, batch_f) in enumerate(train_dataloader):
        #     i = self.item_num

        #     outputs = model(batch_x.to(info.device), token_type_ids=None, attention_mask=batch_m.to(
        #         info.device), labels=batch_y[i + 1].to(info.device))

        #     train_golds[index:index + batch_x.shape[0]] = batch_y[i + 1]

        if mode == 'dev':
            return (qwks, mses), _pred, self.dev_y
        else:
            return (qwks, mses), _pred, self.test_y

    def create_random_attention(self, masks, device):
        random_attn = torch.rand(masks.size())
        random_attn[masks == 0] = 0
        random_attn = random_attn / random_attn.sum(-1).view(-1, 1)
        return random_attn.to(device)

    def create_reversed_attention(self, justification):
        return regularization(1 - justification)

    def evaluate(self, info, model, train_dataloader, print_info=False, all=False, part=True, accuracy_mode=False, is_test=True, attention_type=None, use_loss_for_eval=False):
        assert attention_type in ["random", "reversed", None]
        best = False
        dev_qwks = dict()
        test_qwks = dict()
        dev_mses = dict()
        test_mses = dict()

        dev_score_loss = 0
        test_score_loss = 0
        dev_attn_loss = 0
        test_attn_loss = 0
        dev_grad_loss = 0
        test_grad_loss = 0

        # 指標の計算はcpuで処理
        model.eval()
        with torch.no_grad():
            dev_pred = None
            # dev_attn_output = None
            for batch_x, batch_y, batch_m, batch_a, batch_f in self.dev_data_loader:
                batch_x = batch_x.to(info.device)
                batch_m = batch_m.to(info.device)
                batch_a_ps = batch_a[self.item_num].to(info.device)
                batch_y_ps = batch_y[self.item_num +
                                     1].type(torch.LongTensor).abs().to(info.device)
                if attention_type == "random":
                    justification = self.create_random_attention(
                        batch_m, info.device)
                elif attention_type == "reversed":
                    justification = model(batch_x, token_type_ids=None,
                                          attention_mask=batch_m, labels=batch_y_ps)[1]
                    justification = self.create_reversed_attention(
                        justification)
                elif attention_type is None:
                    justification = None
                else:
                    raise ValueError

                # batch_dev_pred = model(batch_x.to(info.device), token_type_ids=None,
                #                        attention_mask=batch_m.to(info.device), labels=batch_y[self.item_num + 1].to(info.device), justification=random_attn)[0]
                batch_dev_outputs = model(batch_x, token_type_ids=None,
                                          attention_mask=batch_m, labels=batch_y_ps, justification=justification)
                if dev_pred is None:
                    dev_pred = batch_dev_outputs[0]
                    # dev_attn_output = batch_dev_outputs[1]
                else:
                    dev_pred = torch.cat((dev_pred, batch_dev_outputs[0]))
                    # dev_attn_output = torch.cat(
                    #     (dev_attn_output, batch_dev_outputs[1]))

                # score loss計算
                batch_dev_score_loss = self.criterion(
                    batch_dev_outputs[0], batch_y_ps)
                dev_score_loss += batch_dev_score_loss * batch_x.shape[0]

                # attn loss計算
                p_len = batch_dev_outputs[1].shape[1]
                # batch_f = batch_f[self.item_num + 1].to(info.device)
                batch_dev_attn_loss = attn_loss(batch_dev_outputs[1],
                                                batch_a_ps[:, :p_len])
                dev_attn_loss += batch_dev_attn_loss * batch_x.shape[0]

                # grad_lossを計算
                if self.method is not None:
                    with torch.backends.cudnn.flags(enabled=False):
                        interpretable_emb = configure_interpretable_embedding_layer(
                            model, "embedding")
                        model.zero_grad()
                        emb_out = model.get_embeddings(
                            batch_x, attention_mask=batch_m)
                        input_embed = interpretable_emb.indices_to_embeddings(
                            emb_out)
                        attribution = self.method.attribute(
                            inputs=input_embed, target=batch_y_ps, additional_forward_args=(batch_m))
                        assert attribution.size() == input_embed.size()
                        grad_norm = torch.norm(attribution, dim=-1)
                        assert grad_norm.size() == batch_x.size()

                        grad_norm = regularization(grad_norm)
                        assert grad_norm.size() == batch_x.size()

                        batch_grad_loss = attn_loss(
                            grad_norm, batch_a_ps)
                        dev_grad_loss += batch_grad_loss * batch_x.shape[0]
                        remove_interpretable_embedding_layer(
                            model, interpretable_emb)

            test_pred = None
            # test_attn_output = None
            for batch_x, batch_y, batch_m, batch_a, batch_f in self.test_data_loader:
                batch_x = batch_x.to(info.device)
                batch_m = batch_m.to(info.device)
                batch_a_ps = batch_a[self.item_num].to(info.device)
                batch_y_ps = batch_y[self.item_num +
                                     1].type(torch.LongTensor).abs().to(info.device)

                if attention_type == "random":
                    justification = self.create_random_attention(
                        batch_m, info.device)
                elif attention_type == "reversed":
                    justification = model(batch_x, token_type_ids=None,
                                          attention_mask=batch_m, labels=batch_y_ps)[1]
                    justification = self.create_reversed_attention(
                        justification)
                elif attention_type is None:
                    justification = None
                else:
                    raise ValueError
                # batch_test_pred = model(batch_x.to(info.device), token_type_ids=None,
                #                         attention_mask=batch_m.to(info.device), labels=batch_y[self.item_num + 1].to(info.device), justification=random_attn)[0]
                batch_test_outputs = model(
                    batch_x, token_type_ids=None, attention_mask=batch_m, labels=batch_y_ps, justification=justification)
                if test_pred is None:
                    test_pred = batch_test_outputs[0]
                    # test_attn_output = batch_test_outputs[1]
                else:
                    test_pred = torch.cat((test_pred, batch_test_outputs[0]))
                    # test_attn_output = torch.cat(
                    #     (test_attn_output, batch_test_outputs[1]))

                # score loss計算
                batch_test_score_loss = self.criterion(
                    batch_test_outputs[0], batch_y_ps)
                test_score_loss += batch_test_score_loss * batch_x.shape[0]

                # attn loss計算
                p_len = batch_test_outputs[1].shape[1]
                # batch_f = batch_f[self.item_num + 1]
                batch_test_attn_loss = attn_loss(batch_test_outputs[1],
                                                 batch_a_ps[:, :p_len])
                test_attn_loss += batch_test_attn_loss * batch_x.shape[0]

                # grad_lossを計算
                if self.method is not None:
                    with torch.backends.cudnn.flags(enabled=False):
                        # これは最初に一回やればそれでいいのでは
                        interpretable_emb = configure_interpretable_embedding_layer(
                            model, "embedding")
                        model.zero_grad()
                        emb_out = model.get_embeddings(
                            batch_x, attention_mask=batch_m)
                        input_embed = interpretable_emb.indices_to_embeddings(
                            emb_out)
                        attribution = self.method.attribute(
                            inputs=input_embed, target=batch_y_ps, additional_forward_args=(batch_m))
                        assert attribution.size() == input_embed.size()
                        grad_norm = torch.norm(attribution, dim=-1)
                        assert grad_norm.size() == batch_x.size()

                        grad_norm = regularization(grad_norm)
                        assert grad_norm.size() == batch_x.size()

                        batch_grad_loss = attn_loss(
                            grad_norm, batch_a_ps)
                        test_grad_loss += batch_grad_loss * batch_x.shape[0]
                        remove_interpretable_embedding_layer(
                            model, interpretable_emb)

        dev_grads = None
        test_grads = None

        # 勾配計算のためにgradを保存
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

        # dev_loss = self.criterion(dev_outputs[0], dev_y_org)
        # test_loss = self.criterion(test_outputs[0], test_y_org)
        # dev_loss.backward()
        # test_loss.backward()
        # dev_grads = dev_output.grad
        # test_grads = test_output.grad

        # dev_pred = dev_outputs[0]
        # test_pred = test_outputs[0]

        ds, ts = 0.0, 0.

        d_pred = np.argmax(dev_pred.softmax(1).cpu().detach().numpy(), axis=1)
        t_pred = np.argmax(test_pred.softmax(1).cpu().detach().numpy(), axis=1)
        # logger.debug(dev_pred[i].softmax(1).cpu().detach().numpy().max(1))

        if accuracy_mode:
            dev_qwks, test_qwks = self.calc_accuracy(
                d_pred, t_pred, self.item_num + 1)
        else:
            dev_qwks, test_qwks, dev_lwk, test_lwk = self.calc_qwk(
                d_pred, t_pred, self.item_num + 1)

        dev_mses, test_mses = self.calc_mse(
            d_pred, t_pred, self.item_num + 1, self.info.ps_labels[self.item_num][1])

        # dev_qwks[factor] = d
        # test_qwks[factor] = t

        ds += d_pred
        ts += t_pred

        self.dev_pred = ds
        self.test_pred = ts

        train_hidden_states = torch.zeros(info.train_size, self.info.outdim)
        train_golds = torch.zeros(info.train_size)
        index = 0

        for num, (batch_x, batch_y, batch_m, batch_a, batch_f) in enumerate(train_dataloader):
            i = self.item_num

            outputs = model(batch_x.to(info.device), token_type_ids=None, attention_mask=batch_m.to(
                info.device), labels=batch_y[i + 1].to(info.device))

            train_golds[index:index + batch_x.shape[0]] = batch_y[i + 1]

        # lossを計算
        # 採点のロス計算
        dev_score_loss /= len(self.dev_data_loader)
        test_score_loss /= len(self.test_data_loader)
        # dev_score_loss = (self.criterion(
        #     dev_pred, self.dev_y_org[self.item_num + 1].to(info.device)) * len(dev_pred)) / len(self.dev_data_loader)
        # test_score_loss = (self.criterion(
        #     test_pred, self.test_y_org[self.item_num + 1].to(info.device)) * len(test_pred)) / len(self.test_data_loader)
        # attentionのロス計算
        dev_attn_loss /= len(self.dev_data_loader)
        test_attn_loss /= len(self.test_data_loader)
        # gradientのロス計算
        dev_grad_loss /= len(self.dev_data_loader)
        test_grad_loss /= len(self.test_data_loader)

        if use_loss_for_eval:
            self.print_loss_info(dev_score_loss, dev_attn_loss,
                                 dev_grad_loss, self.item_num, type="Dev")
            self.print_loss_info(
                test_score_loss, test_attn_loss, test_grad_loss, self.item_num, type="Test")
        else:
            self.print_info(dev_qwks, dev_mses, self.item_num, type="Dev")
            self.print_info(test_qwks, test_mses, self.item_num, type="Test")

        return (dev_qwks, test_qwks, dev_mses, test_mses, dev_score_loss, test_score_loss, dev_attn_loss, test_attn_loss, dev_grad_loss, test_grad_loss), None, train_golds, (None, None), (None, None), dev_pred, test_pred, self.dev_y, self.test_y, (None, None), (dev_grads, test_grads)
