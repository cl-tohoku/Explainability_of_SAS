from os import sep
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F
import layer
from logging import setLoggerClass, getLogger
from util import logger as L
import torch
import numpy as np
import metric_learning.metrics as metrics
import codecs
logger = getLogger(__name__)


class BiRnnModelForItemScoringERASER(nn.Module):
    def __init__(self, info, config=None):
        super().__init__()
        self.info = info
        if config is not None:
            self.bert = BertModel.from_pretrained(config)
            #     if self.info.char:
            #         self.bert = BertModel(config).from_pretrained(
            #             "cl-tohoku/bert-base-japanese-char-whole-word-masking")
            #     else:
            #         self.bert = BertModel(config).from_pretrained(
            #             "bert-base-uncased")
        elif info.emb_path != None:
            pretrained_weight = self.load_embeddings()
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim).from_pretrained(
                pretrained_weight, freeze=not info.update_embed)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim)

        self.dropout = nn.Dropout(self.info.dropout_prob)
        self.dropout_rnn = nn.Dropout(self.info.rnn_dropout)
        # self.bi_rnn = nn.LSTM(input_size=self.info.emb_dim, hidden_size=self.info.rnn_dim,
        #                       bidirectional=self.info.bidirectional, batch_first=True)
        if self.info.gru:
            logger.info("use gru")
            self.bi_rnn = nn.GRU(input_size=self.info.emb_dim, hidden_size=self.info.rnn_dim,
                                 bidirectional=self.info.bidirectional, num_layers=info.lstm_num, batch_first=True)
        else:
            self.bi_rnn = nn.LSTM(input_size=self.info.emb_dim, hidden_size=self.info.rnn_dim,
                                  bidirectional=self.info.bidirectional, num_layers=info.lstm_num, batch_first=True)

        attn_input_dim = self.info.rnn_dim * \
            2 if self.info.bilstm_process == "concat" else self.info.rnn_dim
        logger.info(f"attention input dim : {attn_input_dim}")

        # try:
        if self.info.implementation == "funayama":
            self.attention = layer.Attention(attn_input_dim, self.info.rnn_dim)
        elif self.info.implementation == "is_attention_interpretable":
            self.attention = layer.AttentionPreviousPaper(
                attn_input_dim, self.info.rnn_dim)
        else:
            logger.error(f"{self.info.implementation} is not exists")
            raise ValueError(f"{self.info.implementation} is not exists")
        # except:
        #     self.attention = layer.Attention(attn_input_dim, self.info.rnn_dim)

        # self.hidden_fc = nn.Linear(attn_input_dim, self.info.outdim)

        # 可視化用の線形層を噛ませる
        if info.metric == "crossentropy":
            self.fc = nn.Linear(attn_input_dim,
                                (self.info.item_max_score + 1))
            self.metric_name = "CrossEntropy"
        elif info.metric == "cosface":
            self.fc = metrics.ArcMarginProduct(
                attn_input_dim, self.info.item_max_score + 1)
            self.metric_name = "CosFace"
        elif info.metric == "arcface":
            self.fc = metrics.ArcMarginProduct(
                attn_input_dim, self.info.item_max_score + 1)
            self.metric_name = "ArcFace"
        elif info.metric == "sphereface":
            self.fc = metrics.SphereProduct(
                attn_input_dim, self.info.item_max_score + 1)
            self.metric_name = "SphereFace"
        elif info.metric.endswith("Triplet Loss"):
            self.fc = nn.Linear(attn_input_dim, attn_input_dim)
            self.metric_name = "Triplet"

        else:
            raise ValueError(f"{info.metric}")

        self.regularize_vectors = False
        self.metric = info.metric

    def compute_bert_outputs(self, embedding_output, attention_mask=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(
                embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.bert.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.bert.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.bert.parameters()).dtype)
        else:
            head_mask = [None] * self.bert.config.num_hidden_layers

        encoder_outputs = self.bert.encoder(embedding_output,
                                            extended_attention_mask,
                                            head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)
        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

    def forward(self, ids, token_type_ids=None, position_ids=None,
                attention_mask=None, labels=None, justification=None, is_create_featuremap=False, after_emb=False):
        embeddings = []
        sentence_len = attention_mask.sum(-1)
        if self.info.BERT:
            # CLS tokenの分を補正+SEP tokenの分
            sentence_len = sentence_len - 2

        if self.info.BERT:

            if is_create_featuremap:
                if after_emb:
                    output = ids
                else:
                    output = self.compute_bert_outputs(
                        ids, attention_mask)[0][:, 1:, :]
            else:
                # max_tokens = 500
                # if position_ids is None or token_type_ids is None:
                #     output = self.bert(
                #         ids[:, :max_tokens], attention_mask=attention_mask[:, :max_tokens])[0]
                #     for batch_index in range(max_tokens, ids.size(1), max_tokens):
                #         # バッチサイズが1になった時に最初の次元に1を追加する
                #         b_ids = ids[:, batch_index:batch_index + max_tokens]
                #         b_attention_mask = attention_mask[:,
                #                                           batch_index:batch_index + max_tokens]
                #         if len(b_ids.size()) < 2:
                #             b_ids = b_ids.unsqueeze(0)
                #             b_attention_mask = b_attention_mask.unsqueeze(0)
                #         b_output = self.bert(b_ids,
                #                              attention_mask=b_attention_mask)
                #         output = torch.cat((output, b_output[0]), dim=1)
                #         del b_output
                # else:
                #     output = self.bert(
                #         ids[:, :max_tokens], attention_mask=attention_mask[:, :max_tokens], position_ids=position_ids[:, :max_tokens], token_type_ids=token_type_ids[:, :max_tokens])[0]
                #     for batch_index in range(max_tokens, ids.size(1), max_tokens):
                #         b_output = self.bert(ids[:, batch_index:batch_index + max_tokens], attention_mask=attention_mask[:, batch_index:batch_index + max_tokens],
                #                              position_ids=position_ids[:, batch_index:batch_index + max_tokens], token_type_ids=token_type_ids[:, batch_index:batch_index + max_tokens])
                #         output = torch.cat((output, b_output[0]), dim=1)
                #         del b_output
                # output = output[:, 1:, :]
                output = self.bert(ids, attention_mask=attention_mask,
                                   position_ids=position_ids, token_type_ids=token_type_ids)[0][:, 1:, :]
        else:
            output = self.embedding(ids)

        embeddings.append(output)

        output = self.dropout(output)

        sentence_len = sentence_len.view(-1).long()

        output = nn.utils.rnn.pack_padded_sequence(
            output, sentence_len, batch_first=True, enforce_sorted=False)
        output, _ = self.bi_rnn(output)
        # logger.debug(f"{output.data.shape}")
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if self.info.bidirectional:
            if self.info.bilstm_process == 'concat':
                None
            elif self.info.bilstm_process == 'sum':
                output = output[:, :, :self.info.rnn_dim] + \
                    output[:, :, self.info.rnn_dim:]
            elif self.info.bilstm_process == 'mean':
                output = (output[:, :, :self.info.rnn_dim] +
                          output[:, :, self.info.rnn_dim:]) / 2
            else:
                raise ValueError(self.info.bilstm_process)
            output = output.contiguous()

        # output = self.dropout_rnn(output)
        if justification is not None:
            weighted_sums, attention_weights = self.attention(
                output, gold_weights=justification)
        else:
            weighted_sums, attention_weights = self.attention(output)

        norm = torch.norm(output, dim=-1).detach()
        del output

        if self.info.metric == "crossentropy" or self.info.metric.endswith("Triplet Loss"):
            target_embeddings = self.fc(weighted_sums)
        else:
            target_embeddings = self.fc(weighted_sums, labels)

        del weighted_sums
        if self.info.regularize_vectors:
            # 正規化
            target_embeddings = target_embeddings / \
                torch.unsqueeze(torch.norm(target_embeddings, dim=-1), dim=-1)

        # dropout
        # target_embeddings = self.dropout_rnn(target_embeddings)

        return target_embeddings, attention_weights, (target_embeddings, None, norm)

    def predict(self, ids, attention_mask=None, token_type_ids=None, position_ids=None, is_create_featuremap=False, after_emb=False):
        # if self.info.BERT:
        #     ids = ids.squeeze(0)
        target_embeddings, _, _ = self.forward(
            ids=ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            is_create_featuremap=is_create_featuremap,
            after_emb=after_emb)
        return target_embeddings

    def get_embeddings(self, ids, token_type_ids=None, position_ids=None, attention_mask=None):
        sentence_len = attention_mask.sum(-1)
        if self.info.BERT:
            # CLS tokenの分を補正+SEP tokenの分
            sentence_len = sentence_len - 2
        if self.info.BERT:
            # max_tokens = 500
            # output = self.bert(
            #     ids[:, :max_tokens], attention_mask=attention_mask[:, :max_tokens])[0]
            # for batch_index in range(max_tokens, ids.size(1), max_tokens):
            #     b_output = self.bert(ids[:, batch_index:batch_index + max_tokens],
            #                          attention_mask=attention_mask[:, batch_index:batch_index + max_tokens])
            #     output = torch.cat((output, b_output[0]), dim=1)
            #     del b_output
            # output = output[:, 1:, :]
            output = self.bert(ids, attention_mask=attention_mask,
                               position_ids=position_ids, token_type_ids=token_type_ids)[0][:, 1:, :]
        else:
            output = self.embedding(ids)
        # output = torch.tensor(output, requires_grad=True)
        return output

    def get_grads(self, output, attention_mask=None, labels=None, justification=None):
        sentence_len = attention_mask.sum(-1)
        if self.info.BERT:
            # CLS tokenの分を補正
            sentence_len = sentence_len - 1
        output = nn.utils.rnn.pack_padded_sequence(
            output, sentence_len, batch_first=True, enforce_sorted=False)
        embeddings = [output]
        output, h = self.bi_rnn(output)
        # logger.debug(f"{output.data.shape}")
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()
            # output = self.dropout_rnn(output)
        # output = self.dropout_rnn(output)

        weighted_sums, attention_weights = self.attention(output)
        # weighted_sums = self.hidden_fc(weighted_sums)
        if self.info.metric == "crossentropy":
            target_embeddings = self.fc(weighted_sums)
        else:
            target_embeddings = self.fc(weighted_sums, labels)
        if self.info.regularize_vectors:
            # 正規化
            target_embeddings = target_embeddings / \
                torch.unsqueeze(torch.norm(target_embeddings, dim=-1), dim=-1)
        return target_embeddings, attention_weights, (weighted_sums, embeddings, output)

    def load_embeddings(self):
        vec_path = self.info.emb_path
        vocab = self.info.vocab
        vocab_size = len(vocab)
        pre_weight = torch.randn(vocab_size, self.info.emb_dim)

        logger.info(f"loading pretrained-vectors from {self.info.emb_path}")

        with codecs.open(vec_path, "r", encoding='utf-8') as fi:
            for i, line in enumerate(fi):
                if i == 0:
                    continue
                line_list = line.strip().split(" ", 1)
                word = line_list[0]
                if word in vocab:
                    # 0ベクトルになってしまう単語あるので除外する
                    value = torch.Tensor(
                        np.array(line.strip().split(" ")[1::], dtype=np.float32))
                    if value.size(0) == self.info.emb_dim:
                        pre_weight[vocab[word]] = value
                    else:
                        logger.warning(
                            f"size mismatch {value.size(0)}, {self.info.emb_dim}")

        # <zero> tokenは0ベクトルを挿入
        pre_weight[vocab['<zero>']] = torch.Tensor(np.zeros_like(
            np.array(line.strip().split(" ")[1::]), dtype=np.float32))

        logger.info("done")
        if not self.info.emb_train:
            logger.info(
                "The embedding layer does not get updated in the learning process.")
        return pre_weight

    def freeze_bert_pram(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        logger.info(f"bert params was freezed")

    def unfreeze_bert_param(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_layer_param(self, layer):
        for param in self.bert.encoder.layer[layer].parameters():
            param.requires_grad = True
        logger.info(f"{layer} layer was unfreezed")

    def print_model_info(self):
        logger.info(f"Train model with {self.metric_name}")


class BertFinetuningForItemScoringERASER(BiRnnModelForItemScoringERASER):
    def __init__(self, info, config):
        super(BiRnnModelForItemScoringERASER, self).__init__()
        self.info = info

        self.bert = BertModel.from_pretrained(config)

        self.dropout = nn.Dropout(self.info.dropout_prob)
        # self.dropout_rnn = nn.Dropout(self.info.rnn_dropout)

        # self.relu = nn.ReLU()
        # self.linear = nn.Linear(self.info.emb_dim, self.info.rnn_dim)

        attn_input_dim = self.bert.config.hidden_size

        if self.info.implementation in ["funayama", "no_lstm_bert"]:
            self.attention = layer.Attention(attn_input_dim, self.info.rnn_dim)
        elif self.info.implementation == "is_attention_interpretable":
            self.attention = layer.AttentionPreviousPaper(
                attn_input_dim, self.info.rnn_dim)
        else:
            logger.error(f"{self.info.implementation} is not exists")
            raise ValueError(f"{self.info.implementation} is not exists")

        # self.fc = nn.Linear(attn_input_dim,
        #                     (self.info.item_max_score + 1))
        # 可視化用の線形層を噛ませる
        if info.metric == "crossentropy":
            self.fc = nn.Linear(attn_input_dim,
                                (self.info.item_max_score + 1))
            self.metric_name = "CrossEntropy"
        elif info.metric == "cosface":
            self.fc = metrics.ArcMarginProduct(
                attn_input_dim, self.info.item_max_score + 1)
            self.metric_name = "CosFace"
        elif info.metric == "arcface":
            self.fc = metrics.ArcMarginProduct(
                attn_input_dim, self.info.item_max_score + 1)
            self.metric_name = "ArcFace"
        elif info.metric == "sphereface":
            self.fc = metrics.SphereProduct(
                attn_input_dim, self.info.item_max_score + 1)
            self.metric_name = "SphereFace"
        elif info.metric.endswith("Triplet Loss"):
            self.fc = nn.Linear(attn_input_dim, attn_input_dim)
            self.metric_name = "Triplet"
        else:
            raise ValueError(f"{info.metric}")

        self.metric_name = "CrossEntropy"

    def compute_bert_outputs(self, embedding_output, attention_mask=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(
                embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.bert.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.bert.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.bert.parameters()).dtype)
        else:
            head_mask = [None] * self.bert.config.num_hidden_layers

        encoder_outputs = self.bert.encoder(embedding_output,
                                            extended_attention_mask,
                                            head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)
        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

    def forward(self, ids, token_type_ids=None, position_ids=None,
                attention_mask=None, labels=None, justification=None, is_create_featuremap=False, after_emb=False):
        embeddings = []
        sentence_len = attention_mask.sum(-1)

        sentence_len = sentence_len - 2

        if is_create_featuremap:
            if after_emb:
                output = ids
            else:
                output = self.compute_bert_outputs(
                    ids, attention_mask)[0][:, 1:, :]
        else:
            output = self.bert(ids, attention_mask=attention_mask,
                               position_ids=position_ids, token_type_ids=token_type_ids)[0][:, 1:, :]

        output = nn.utils.rnn.pack_padded_sequence(
            output, sentence_len, batch_first=True, enforce_sorted=False)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # # [PAD]部分を0埋め
        # output[attention_mask[:, 1:] == 0] = 0.0

        # # SEP以降を0埋め
        # output[:, :-1, :][attention_mask[:, 2:] == 0] = 0.0
        # # SEP分を削除
        # output = output[:, :-1, :]

        # print(attention_mask.sum(-1))

        embeddings.append(output)
        # print(output[:, attention_mask.sum(1)])

        output = self.dropout(output)

        # output, _ = self.attention(outpust)

        # output = self.linear(output)
        # output = self.relu(output)

        if justification is not None:
            weighted_sums, attention_weights = self.attention(
                output, gold_weights=justification)
        else:
            weighted_sums, attention_weights = self.attention(output)

        norm = torch.norm(output, dim=-1).detach()
        del output

        target_embeddings = self.fc(weighted_sums)

        del weighted_sums

        if self.info.regularize_vectors:
            # 正規化
            target_embeddings = target_embeddings / \
                torch.unsqueeze(torch.norm(target_embeddings, dim=-1), dim=-1)

        return target_embeddings, attention_weights, (None, None, norm)

    # def unfreeze_bert_param(self, layers=[-1]):
    #     for layer_num in layers:
    #         for param in self.bert.encoder.layer[layer_num].parameters():
    #             param.requires_grad = True

    def get_embeddings(self, ids, token_type_ids=None, position_ids=None, attention_mask=None):

        output = self.bert(ids, attention_mask=attention_mask,
                           position_ids=position_ids, token_type_ids=token_type_ids)[0][:, 1:, :]
        return output

    def predict(self, ids, attention_mask=None, token_type_ids=None, position_ids=None, is_create_featuremap=False, after_emb=False):
        # if self.info.BERT:
        #     ids = ids.squeeze(0)
        target_embeddings, _, _ = self.forward(
            ids=ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            is_create_featuremap=is_create_featuremap,
            after_emb=after_emb)
        return target_embeddings

    def freeze_bert_pram(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        logger.info(f"bert params was freezed")

    def unfreeze_bert_param(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_layer_param(self, layer):
        for param in self.bert.encoder.layer[layer].parameters():
            param.requires_grad = True
        logger.info(f"{layer} layer was unfreezed")
