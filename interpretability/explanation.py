import torch
import logging
from captum.attr import *
from tqdm import tqdm
from logging import getLogger
import pandas as pd
import sys
from interpretability.metric_captum import MetricSaliency, MetricInputXGradient, MetricIntegratedGradients
from make_knn_xlsx import load_pickle
logger = getLogger(__name__)


class Explanation:
    def __init__(self, model, settings, debug=False):
        self.settings = settings
        self.net = model
        self.net.to(self.settings.device)
        self.net.normalization = False
        self.debug = debug

    def _calc_attn(self, ids, attention_mask, labels):
        # ids = ids.to(self.settings.device)
        # attention_mask = attention_mask.to(self.settings.device)
        maps = []

        bar = tqdm(total=len(ids))
        self.net.eval()
        with torch.no_grad():
            for id, a_m, label in tqdm(zip(ids, attention_mask, labels)):
                id = id[:a_m.sum(-1)]
                id = id.unsqueeze(0).to(self.settings.device)
                a_m = a_m[:a_m.sum(-1)].unsqueeze(0).to(self.settings.device)
                # _, attn, _ = self.net(source.view(1, -1), batch_size=1)
                _, attention_weights, _ = self.net(ids=id, attention_mask=a_m)

                maps.append(attention_weights.squeeze(
                    0).detach().to("cpu").numpy().tolist())
                bar.update(1)
        return maps

    def _calc_random(self, ids, attention_mask):
        maps = []
        for id, a_m, in tqdm(zip(ids, attention_mask)):

            if self.settings.BERT:
                # CLSとSEP以降を削除
                id = id[1:a_m.sum(-1) - 1]
            else:
                id = id[:a_m.sum(-1)]
            rand_map = torch.rand(size=id.size(),
                                  device=self.settings.device)
            rand_map = self.regularization(rand_map)
            maps.append(rand_map.squeeze(
                0).detach().to("cpu").numpy().tolist())

        return maps

    def _calc_norm(self, ids, attention_mask, labels):
        # ids = ids.to(self.settings.device)
        maps = []
        bar = tqdm(total=len(ids))
        self.net.eval()
        with torch.no_grad():
            for id, a_m, label in tqdm(zip(ids, attention_mask, labels)):
                id = id[:a_m.sum(-1)]
                id = id.unsqueeze(0).to(self.settings.device)
                a_m = a_m[:a_m.sum(-1)].unsqueeze(0).to(self.settings.device)
                _, _, (_, _, norm) = self.net(ids=id, attention_mask=a_m)
                norm = self.regularization(norm)
                maps.append(norm.squeeze(0).detach().to(
                    "cpu").numpy().tolist())
                bar.update(1)
        return maps

    def _calc_norm_attn(self, ids, attention_mask, labels):
        maps = []
        bar = tqdm(total=len(ids))
        self.net.eval()
        with torch.no_grad():
            for id, a_m, label in tqdm(zip(ids, attention_mask, labels)):
                id = id[:a_m.sum(-1)]
                id = id.unsqueeze(0).to(self.settings.device)
                a_m = a_m[:a_m.sum(-1)].unsqueeze(0).to(self.settings.device)
                _, attention_weights, (_, _, norm) = self.net(
                    ids=id, attention_mask=a_m)
                norm_attn = norm * attention_weights
                norm_attn = self.regularization(norm_attn)
                maps.append(norm_attn.squeeze(
                    0).detach().to("cpu").numpy().tolist())
                bar.update(1)
        return maps

    def regularization(self, target_embeddings):
        _target = torch.clone(target_embeddings)
        return _target / torch.sum(_target)

    def _generate_explanation(self, ids, attention_mask, labels, method, n_samples=None, nt_type=None, nt_samples=None, knn_model=None):

        smoothgrad = True if nt_type is not None and nt_samples is not None else False

        grad_maps = []

        if self.settings.BERT:
            None
        else:
            interpretable_emb = configure_interpretable_embedding_layer(
                self.net, "embedding")

        self.net.train()

        bar = tqdm(total=len(ids))
        # for source, target in tqdm(zip(sources, targets)):
        for id, label, a_m in zip(ids, labels, attention_mask):
            a_m = a_m.unsqueeze(0).to(self.settings.device)
            id = id.to(self.settings.device)
            self.net.zero_grad()
            m = method(self.net.predict) if knn_model is None else method(
                self.net.predict, knn_model)
            if smoothgrad:
                m = NoiseTunnel(m)

            # ここをw2vとbertで変わるように関数を書く

            if self.settings.BERT:
                if self.settings.after_emb:
                    input_embed = self.net.get_embeddings(
                        id.unsqueeze(0), attention_mask=a_m)
                    token_type_ids_embed = None
                    position_ids_embed = None
                else:
                    self.net.zero_grad()
                    # token_type_ids_embed = torch.zeros_like(
                    #     id).type(torch.LongTensor).unsqueeze(0).to(self.settings.device)
                    # position_ids_embed = torch.zeros_like(
                    #     id).type(torch.LongTensor).unsqueeze(0).to(self.settings.device)
                    token_type_ids_embed = None
                    position_ids_embed = None

                    input_embed = self.net.bert.embeddings(id.unsqueeze(0))

                if smoothgrad:
                    attribution = m.attribute(inputs=input_embed, nt_type=nt_type, nt_samples=nt_samples,
                                              target=label, additional_forward_args=(a_m, token_type_ids_embed, position_ids_embed))
                else:
                    attribution = m.attribute(inputs=input_embed, target=label, additional_forward_args=(
                        a_m, token_type_ids_embed, position_ids_embed, True, self.settings.after_emb))

            else:
                input_emb = interpretable_emb.indices_to_embeddings(id)
                # unsqueezeしないとrnnあたりで変になる
                input_emb = input_emb.unsqueeze(0).to(self.settings.device)
                if smoothgrad:
                    attribution = m.attribute(
                        inputs=input_emb, target=label, nt_type=nt_type, nt_samples=nt_samples, additional_forward_args=(a_m))
                if n_samples is None:
                    attribution = m.attribute(
                        inputs=input_emb, target=label, additional_forward_args=(a_m))
                else:
                    attribution = m.attribute(
                        inputs=input_emb, target=label, additional_forward_args=(a_m), n_samples=n_samples)

            # 出力をmaskの大きさにする
            attribution = attribution.squeeze(0)
            if self.settings.BERT:
                # CLSとSEP以降を削除
                attribution = attribution[1:a_m.sum(-1) - 1]
            else:
                attribution = attribution[:a_m.sum(-1)]

            grad_map = torch.norm(attribution, dim=-1)
            grad_map = self.regularization(grad_map)
            grad_maps.append(grad_map.detach().to("cpu").numpy().tolist())
            del grad_map
            bar.update(1)

        if self.settings.BERT:
            None
        else:
            remove_interpretable_embedding_layer(self.net, interpretable_emb)

        return grad_maps

    def _calc_explanation_sequentially(self, ids, attention_mask, labels):
        # yield "Lime", self._generate_explanation(ids, attention_mask, labels, Lime, 1)
        # yield "DeepLift", self._generate_explanation(ids, attention_mask, labels, DeepLift)
        # yield "DeepLiftShap", self._generate_explanation(ids, attention_mask, labels, DeepLiftShap)
        yield "Random", self._calc_random(ids, attention_mask)
        yield "Attention_Weights", self._calc_attn(ids, attention_mask, labels)
        yield "Saliency", self._generate_explanation(ids, attention_mask, labels, Saliency)
        # yield "Norm", self._calc_norm(ids, attention_mask, labels)
        # yield "Norm*Attention_Weights", self._calc_norm_attn(ids, attention_mask, labels)
        yield "Integrated_Gradients", self._generate_explanation(ids, attention_mask, labels, IntegratedGradients)
        yield "Input_X_Gradient", self._generate_explanation(ids, attention_mask, labels, InputXGradient)

        # yield "SmoothGrad", self._generate_explanation(ids, attention_mask, labels, Saliency, 'smoothgrad', 10)
        # yield "Feature_Ablation", self._generate_explanation(ids, attention_mask, labels, FeatureAblation)

    def _calc_explanation_sequentially_instancebase(self, ids, attention_mask, labels, knn_model):
        yield "Integrated_Gradients", self._generate_explanation(ids, attention_mask, labels, MetricIntegratedGradients, knn_model=knn_model)
        yield "Saliency", self._generate_explanation(ids, attention_mask, labels, MetricSaliency, knn_model=knn_model)
        yield "Random", self._calc_random(ids, attention_mask)
        yield "Input_X_Gradient", self._generate_explanation(ids, attention_mask, labels, MetricInputXGradient, knn_model=knn_model)
        yield "Attention_Weights", self._calc_attn(ids, attention_mask, labels)
        yield "Norm", self._calc_norm(ids, attention_mask, labels)
        yield "Norm*Attention_Weights", self._calc_norm_attn(ids, attention_mask, labels)

    def _generate_predicted_labels(self, sources):
        self.net.to(self.settings.device)

        targets = []
        self.net.eval()
        with torch.no_grad():
            for source in tqdm(sources):
                target = self.net(source.unsqueeze(0).to(
                    self.settings.device), batch_size=1)
                targets.append(int(torch.argmax(target, dim=1).detach().cpu()))

        return targets

    def _limitation(self, sequence, limit):
        if self.debug:
            logging.debug("debug mode")
            return sequence[:10]
        elif limit is not None:
            return sequence[:limit]
        else:
            return sequence

    def explanation_for_instacebase(self, ids, attention_mask, labels, info, prediction=False, limit=None, ):
        # attention_mask = attention_mask.to(self.settings.device)
        # explanation_df = pd.DataFrame()

        # explanation_df["Attention_Weights"] = self._calc_attn(
        #     ids, attention_mask, labels)
        # explanation_df["Norm"] = self._calc_norm(ids, attention_mask, labels)

        from interpretability.knn_model import KnnModel
        train_data, train_label, * \
            _ = load_pickle(f"{info.out_dir}_train_result.pickle")
        knn_model = KnnModel(train_data, train_label)

        explanation_df = pd.DataFrame()
        map_size = None
        for explanation_name, maps in self._calc_explanation_sequentially_instancebase(ids, attention_mask, labels, knn_model):
            assert map_size is None or len(
                maps[0]) == map_size, f"{map_size}, {len(maps[0])}"
            map_size = len(maps[0])
            logging.info(
                "Processing of {} was finished.".format(explanation_name))
            explanation_df[explanation_name] = maps
        return explanation_df

    def __call__(self, ids, attention_mask, labels, prediction=False, limit=None):
        # sources = self._limitation(dataset["sources"], limit)

        # 普通にやるならgold labelを使う
        # if prediction:
        #     labels = self._generate_predicted_labels(ids, attention_mask)

        # ids = ids.to(self.settings.device)
        # attention_mask = attention_mask.to(self.settings.device)
        explanation_df = pd.DataFrame()
        map_size = None
        for explanation_name, maps in self._calc_explanation_sequentially(ids, attention_mask, labels):
            assert map_size is None or len(
                maps[0]) == map_size, f"{map_size}, {len(maps[0])}, {len(ids[0])}, {ids[0]}"
            map_size = len(maps[0])
            logging.info(
                "Processing of {} was finished.".format(explanation_name))
            explanation_df[explanation_name] = maps
        return explanation_df
