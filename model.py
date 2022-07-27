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


class BertForScoring(BertPreTrainedModel):
    def __init__(self, config, t_info):
        super(BertForScoring, self).__init__(config)
        self.logging_model_info()
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            config.hidden_size, self.config.num_labels)

        self.attention = layer.Attention(config.hidden_size)

    def forward(self, ids, token_type_ids=None,
                attention_mask=None, labels=None):
        output = self.bert(ids, token_type_ids, attention_mask)

        output = output[2][11][:,  1:, :]
        output, _ = self.attention(output)
        output = self.dropout(output)
        logits = self.classifier(output)

        return logits

    def freeze_bert_pram(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_param(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_layer_param(self, layer):
        for param in self.bert.encoder.layer[layer].parameters():
            param.requires_grad = True

    def logging_model_info(self):
        logger.info("Build Bert model with Simple Classifier.")

    def get_module_group(self):
        predictor = []
        for module in self.modules():
            if isinstance(module, BertModel):
                bertmodule = module
            else:
                predictor.append(module)
        return predictor, bertmodule


class BertForScoringWithLSTM(BertForScoring):
    def __init__(self, config, t_info):
        super(BertForScoringWithLSTM, self).__init__(config, t_info)
        self.info = t_info
        self.bi_rnn = nn.LSTM(input_size=768, hidden_size=self.info.rnn_dim,
                              bidirectional=self.info.bidirectional, batch_first=True)
        self.attention = layer.Attention(self.info.rnn_dim)
        self.classifier = nn.Linear(
            self.info.rnn_dim, self.config.num_labels)
        self.linear = nn.Linear(
            self.info.rnn_dim, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_rnn = nn.Dropout(self.info.dropout_prob)
        if self.info.reg:
            logger.info("Build Regression model.")
        else:
            logger.info("Build Classification model.")

    def forward(self, ids, token_type_ids=None,
                attention_mask=None, labels=None):
        output = self.bert(ids, token_type_ids, attention_mask)
        embeddings = output
        output = output[0][:, 1:, :]

        output = self.dropout(output)
        output = nn.utils.rnn.pack_padded_sequence(
            output,  attention_mask.sum(1) - 2, batch_first=True, enforce_sorted=False)

        output, h = self.bi_rnn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()

        output = self.dropout_rnn(output)

        if self.info.mot:
            # output = self.dropout(output)
            m = attention_mask.sum(1).type(
                torch.FloatTensor).to(self.info.device)
            # MoT
            output = output.sum(1) / torch.unsqueeze(m,
                                                     dim=1).expand(m.shape[0], self.info.rnn_dim)
        else:
            output, weight = self.attention(output)
            # output = output.mean(1)
        hidden_states = output
        if self.info.reg:
            output = self.linear(output).squeeze()
        else:
            output = self.classifier(output)

        return output, None, (hidden_states, embeddings)

    def logging_model_info(self):
        logger.info("Build Bert model with LSTM.")


class BertForPartScoring(BertPreTrainedModel):
    def __init__(self, config, t_info):
        super(BertForPartScoring, self).__init__(config, t_info)
        logger.info("Build Bert model with Classifier.")
        self.info = t_info
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(p=0.5)
        self.attention = layer.Attention(self.info.rnn_dim)

        self.fc = nn.ModuleList()
        self.attention = nn.ModuleList()

        for i, label in enumerate(t_info.ps_labels):
            self.fc.append(nn.Linear(
                config.hidden_size, label[1] + 1))

    def forward(self, ids, token_type_ids=None,
                attention_mask=None, labels=None):
        output = self.bert(ids, token_type_ids, attention_mask)
        output = output[0][:,  1:, :].mean(1)
        # output, _ = self.attention(output)
        output = self.dropout(output)

        scores = []
        for i, label in enumerate(self.info.ps_labels):
            output_fc = self.fc[i](output)
            scores.append(output_fc)

        return scores

    def logging_model_info(self):
        logger.info("Build Bert model for part scoring with Simple Classifier.")


class BertForPartScoringWithLSTM(nn.Module):
    def __init__(self, config, info):
        super(BertForPartScoringWithLSTM, self).__init__()
        self.info = info
        # self.num_labels = config.num_labels
        if self.info.char:
            self.bert = BertModel(config).from_pretrained(
                "bert-base-japanese-char")
        else:
            self.bert = BertModel(config).from_pretrained(
                "bert-base-japanese-whole-word-masking")
        self.dropout = nn.Dropout(0.1)
        self.dropout_rnn = nn.Dropout(self.info.dropout_prob)
        self.bi_rnn = nn.LSTM(input_size=config.hidden_size, hidden_size=self.info.rnn_dim,
                              bidirectional=self.info.bidirectional, batch_first=True)
        self.attention = layer.Attention(self.info.rnn_dim)
        self.fc = nn.ModuleList()
        self.attention = nn.ModuleList()
        # self.dmy = nn.Parameter(torch.FloatTensor(768))
        # self.dmy.requires_grad = False

        for i, label in enumerate(self.info.ps_labels):
            self.attention.append(layer.Attention(self.info.rnn_dim))
            self.fc.append(nn.Linear(
                self.info.rnn_dim, label[1] + 1))

    def forward(self, ids, token_type_ids=None,
                attention_mask=None, labels=None):

        embeddings = []
        # print(attention_mask.sum(1))

        output = self.bert(ids, token_type_ids, attention_mask)

        output = output[0][:,  1:, :]

        embeddings.append(output)

        # print(output[:, attention_mask.sum(1)])

        # output, _ = self.attention(outpust)
        output = self.dropout(output)
        output = nn.utils.rnn.pack_padded_sequence(
            output,  attention_mask.sum(1) - 1, batch_first=True, enforce_sorted=False)

        output, h = self.bi_rnn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()
            # output = self.dropout_rnn(output)

        hidden_states = []
        scores = []
        attentions = []
        for i, label in enumerate(self.info.ps_labels):
            output_attn, attention_weight = self.attention[i](output)
            output_fc = self.fc[i](output_attn)
            scores.append(output_fc)
            attentions.append(attention_weight)
            hidden_states.append(output_attn)

        return scores, attentions, (hidden_states, embeddings)

    def predict(self, ids, token_type_ids=None,
                attention_mask=None):
        output = self.bert(ids, token_type_ids, attention_mask)

        output = output[0][:,  1:, :]

        embeddings = []
        embeddings.append(output)

        output, h = self.bi_rnn(output)

        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()
            # output = self.dropout_rnn(output)

        hidden_states = []
        scores = []
        attentions = []
        probs = []
        for i, label in enumerate(self.info.ps_labels):
            output_attn, attention_weight = self.attention[i](output)
            output_fc = self.fc[i](output_attn)
            probs.append(output_fc.softmax(1))
            scores.append(output_fc.argmax(1))
            attentions.append(attention_weight)
            hidden_states.append(output_attn)

        return scores, probs, attentions, (hidden_states, embeddings)

    def logging_model_info(self):
        logger.info("Build Bert model for part scoring with LSTM.")

    def freeze_bert_pram(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_param(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_layer_param(self, layer):
        for param in self.bert.encoder.layer[layer].parameters():
            param.requires_grad = True


class BiRnnModel(nn.Module):
    def __init__(self, info):
        super().__init__()

        self.info = info

        if info.emb_path != None:
            pretrained_weight = self.load_embeddings()
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim).from_pretrained(
                pretrained_weight, freeze=not info.update_embed)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim)

        self.dropout = nn.Dropout(0.1)
        self.dropout_rnn = nn.Dropout(self.info.dropout_prob)
        self.bi_rnn = nn.LSTM(input_size=self.info.emb_dim, hidden_size=self.info.rnn_dim,
                              bidirectional=self.info.bidirectional, batch_first=True)
        self.attention = layer.Attention(self.info.rnn_dim)
        self.fc = nn.ModuleList()
        self.attention = nn.ModuleList()
        # self.dmy = nn.Parameter(torch.FloatTensor(768))
        # self.dmy.requires_grad = False

        for i, dim in enumerate(self.info.output_dim):
            self.attention.append(layer.Attention(self.info.rnn_dim))
            self.fc.append(nn.Linear(self.info.rnn_dim, dim))

    def forward(self, ids, token_type_ids=None,
                attention_mask=None, labels=None):

        embeddings = []
        # print(attention_mask.sum(1))

        output = self.embedding(ids)

        embeddings.append(output)

        # print(output[:, attention_mask.sum(1)])

        # output, _ = self.attention(outpust)
        output = self.dropout(output)
        output = nn.utils.rnn.pack_padded_sequence(
            output,  attention_mask.sum(1), batch_first=True, enforce_sorted=False)

        output, h = self.bi_rnn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()
            # output = self.dropout_rnn(output)
        output = self.dropout_rnn(output)
        hidden_states = []
        scores = []
        attentions = []
        for i, label in enumerate(self.info.ps_labels):
            output_attn, attention_weight = self.attention[i](output)
            output_fc = self.fc[i](output_attn)
            scores.append(output_fc)
            attentions.append(attention_weight)
            hidden_states.append(output_attn)

        return scores, attentions, (hidden_states, embeddings)

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
                    value = line.strip().split(" ")[1::]
                    pre_weight[vocab[word]] = torch.Tensor(
                        np.array(value, dtype=np.float32))
        logger.info("done")
        if not self.info.emb_train:
            logger.info(
                "The embedding layer does not get updated in the learning process.")
        return pre_weight


class BiRnnModelForMetric(nn.Module):
    def __init__(self, info):
        super().__init__()

        self.info = info

        if info.emb_path != None:
            pretrained_weight = self.load_embeddings()
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim).from_pretrained(
                pretrained_weight, freeze=not info.update_embed)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim)

        self.dropout = nn.Dropout(0.1)
        self.dropout_rnn = nn.Dropout(self.info.dropout_prob)
        self.bi_rnn = nn.LSTM(input_size=self.info.emb_dim, hidden_size=self.info.rnn_dim,
                              bidirectional=self.info.bidirectional, batch_first=True)
        self.attention = layer.Attention(self.info.rnn_dim)
        self.fc = nn.ModuleList()
        self.attention = nn.ModuleList()
        # self.dmy = nn.Parameter(torch.FloatTensor(768))
        # self.dmy.requires_grad = False

        for i, dim in enumerate(self.info.output_dim):
            self.attention.append(layer.Attention(self.info.rnn_dim))
            self.fc.append(nn.Linear(self.info.rnn_dim, self.info.rnn_dim))

    def forward(self, ids, token_type_ids=None,
                attention_mask=None, labels=None):

        embeddings = []

        output = self.embedding(ids)

        embeddings.append(output)

        # print(output[:, attention_mask.sum(1)])

        # output, _ = self.attention(outpust)
        output = self.dropout(output)
        sentence_len = attention_mask.sum(1)

        output = nn.utils.rnn.pack_padded_sequence(
            output,  sentence_len, batch_first=True, enforce_sorted=False)

        output, h = self.bi_rnn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()
            # output = self.dropout_rnn(output)
        output = self.dropout_rnn(output)
        hidden_states = []
        target_states = []
        attentions = []
        for i, label in enumerate(self.info.ps_labels):
            output_attn, attention_weight = self.attention[i](output)
            fc_output = self.fc[i](output_attn)
            target_states.append(fc_output)
            attentions.append(attention_weight)
            hidden_states.append(output_attn)

        return target_states, attentions, (hidden_states, embeddings)

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
                    value = line.strip().split(" ")[1::]
                    pre_weight[vocab[word]] = torch.Tensor(
                        np.array(value, dtype=np.float32))
        logger.info("done")
        if not self.info.emb_train:
            logger.info(
                "The embedding layer does not get updated in the learning process.")
        return pre_weight

    def predict(self, ids, token_type_ids=None, attention_mask=None):

        embeddings = []
        # print(attention_mask.sum(1))

        output = self.embedding(ids)

        embeddings.append(output)

        # print(output[:, attention_mask.sum(1)])

        # output, _ = self.attention(outpust)
        output = self.dropout(output)
        output = nn.utils.rnn.pack_padded_sequence(
            output,  attention_mask.sum(1), batch_first=True, enforce_sorted=False)

        output, h = self.bi_rnn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()
            # output = self.dropout_rnn(output)
        output = self.dropout_rnn(output)
        hidden_states = []
        scores = []
        attentions = []
        for i, label in enumerate(self.info.ps_labels):
            output_attn, attention_weight = self.attention[i](output)
            output_fc = self.fc[i](output_attn)
            scores.append(output_fc)
            attentions.append(attention_weight)
            hidden_states.append(output_attn.detach())

        return scores, attentions, (hidden_states, embeddings)


class BiRnnModelForItemScoring(nn.Module):
    def __init__(self, info, config=None):
        super().__init__()
        self.info = info
        if config is not None:
            if self.info.char:
                self.bert = BertModel(config).from_pretrained(
                    "cl-tohoku/bert-base-japanese-char-whole-word-masking")
            else:
                self.bert = BertModel(config).from_pretrained(
                    "bert-base-japanese-whole-word-masking")
        elif info.emb_path != None:
            pretrained_weight = self.load_embeddings()
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim).from_pretrained(
                pretrained_weight, freeze=not info.update_embed)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim)

        self.dropout = nn.Dropout(0.1)
        self.dropout_rnn = nn.Dropout(self.info.dropout_prob)
        self.bi_rnn = nn.LSTM(input_size=self.info.emb_dim, hidden_size=self.info.rnn_dim,
                              bidirectional=self.info.bidirectional, batch_first=True)
        self.attention = layer.Attention(self.info.rnn_dim)
        # self.dmy = nn.Parameter(torch.FloatTensor(768))
        # self.dmy.requires_grad = False
        self.attention = layer.Attention(self.info.rnn_dim)
        # assert info.metric in {"crossentropy", "cosface", "arcface",
        #                        "sphereface"}, f"{info.metric} is not valid metric."

        # 可視化用の線形層を噛ませる
        self.hidden_fc = nn.Linear(self.info.rnn_dim, self.info.outdim)
        if info.metric == "crossentropy":
            self.fc = nn.Linear(self.info.outdim,
                                (self.info.item_max_score + 1))
            self.metric_name = "CrossEntropy"
        elif info.metric == "cosface":
            self.fc = metrics.ArcMarginProduct(
                self.info.outdim, self.info.item_max_score + 1)
            self.metric_name = "CosFace"
        elif info.metric == "arcface":
            self.fc = metrics.ArcMarginProduct(
                self.info.outdim, self.info.item_max_score + 1)
            self.metric_name = "ArcFace"
        elif info.metric == "sphereface":
            self.fc = metrics.SphereProduct(
                self.info.outdim, self.info.item_max_score + 1)
            self.metric_name = "SphereFace"
        self.regularize_vectors = False
        self.metric = info.metric

    def forward(self, ids, token_type_ids=None, position_ids=None,
                attention_mask=None, labels=None, justification=None):
        embeddings = []

        sentence_len = attention_mask.sum(-1)
        if self.info.BERT:
            # CLS tokenの分を補正+SEP tokenの分
            sentence_len = sentence_len - 2

        if self.info.BERT:
            output = self.bert(ids, token_type_ids=token_type_ids,
                               position_ids=position_ids, attention_mask=attention_mask)
            output = output[0][:,  1:, :]
        else:
            output = self.embedding(ids)

        embeddings.append(output)
        # print(output[:, attention_mask.sum(1)])

        # output, _ = self.attention(outpust)
        # output = self.dropout(output))
        output = nn.utils.rnn.pack_padded_sequence(
            output, sentence_len, batch_first=True, enforce_sorted=False)
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

    def predict(self, ids, attention_mask=None, token_type_ids=None, position_ids=None):
        # if self.info.BERT:
        #     ids = ids.squeeze(0)
        target_embeddings, _, _ = self.forward(
            ids=ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        return target_embeddings

    def get_embeddings(self, ids, token_type_ids=None, attention_mask=None):
        if self.info.BERT:
            output = self.bert(ids, token_type_ids, attention_mask)
            output = output[0][:,  1:, :]
        else:
            output = self.embedding(ids)
        output = torch.tensor(output, requires_grad=True)
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
        logger.info("done")
        if not self.info.emb_train:
            logger.info(
                "The embedding layer does not get updated in the learning process.")
        return pre_weight

    def freeze_bert_pram(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_param(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_layer_param(self, layer):
        for param in self.bert.encoder.layer[layer].parameters():
            param.requires_grad = True

    def print_model_info(self):
        logger.info(f"Train model with {self.metric_name}")


class BiRnnModelForItemScoringWithMetric(nn.Module):
    def __init__(self, info, config=None):
        super().__init__()
        self.info = info
        if config is not None:
            if self.info.char:
                self.bert = BertModel(config).from_pretrained(
                    "cl-tohoku/bert-base-japanese-char-whole-word-masking")
            else:
                self.bert = BertModel(config).from_pretrained(
                    "bert-base-japanese-whole-word-masking")
        elif info.emb_path != None:
            pretrained_weight = self.load_embeddings()
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim).from_pretrained(
                pretrained_weight, freeze=not info.update_embed)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim)

        self.dropout = nn.Dropout(0.1)
        self.dropout_rnn = nn.Dropout(self.info.dropout_prob)
        self.bi_rnn = nn.LSTM(input_size=self.info.emb_dim, hidden_size=self.info.rnn_dim,
                              bidirectional=self.info.bidirectional, batch_first=True)
        self.attention = layer.Attention(self.info.rnn_dim)
        # self.dmy = nn.Parameter(torch.FloatTensor(768))
        # self.dmy.requires_grad = False
        self.attention = layer.Attention(self.info.rnn_dim)
        # self.fc = nn.Linear(self.info.rnn_dim, self.info.rnn_dim)
        self.fc = nn.Linear(self.info.rnn_dim, self.info.outdim)
        self.regularize_vectors = False

    def forward(self, ids, token_type_ids=None, position_ids=None,
                attention_mask=None, labels=None, justification=None):
        embeddings = []

        sentence_len = attention_mask.sum(-1)
        if self.info.BERT:
            # CLS tokenの分を補正+SEP tokenの分
            sentence_len = sentence_len - 2

        if self.info.BERT:
            output = self.bert(ids, token_type_ids=token_type_ids,
                               position_ids=position_ids, attention_mask=attention_mask)
            output = output[0][:,  1:, :]
        else:
            output = self.embedding(ids)

        embeddings.append(output)

        # print(output[:, attention_mask.sum(1)])

        # output, _ = self.attention(outpust)
        # output = self.dropout(output)
        output = nn.utils.rnn.pack_padded_sequence(
            output, sentence_len, batch_first=True, enforce_sorted=False)
        output, h = self.bi_rnn(output)
        # logger.debug(f"{output.data.shape}")
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()
            # output = self.dropout_rnn(output)
        # output = self.dropout_rnn(output)

        if justification is not None:
            weighted_sums, attention_weights = self.attention(
                output, gold_weights=justification)
        else:
            weighted_sums, attention_weights = self.attention(output)

        target_embeddings = self.fc(weighted_sums)
        if self.info.regularize_vectors:
            # 正規化
            if self.info.affin:
                target_embeddings = target_embeddings / \
                    torch.unsqueeze(torch.norm(
                        target_embeddings, dim=-1), dim=-1)
            else:
                weighted_sums = weighted_sums / \
                    torch.unsqueeze(torch.norm(weighted_sums, dim=-1), dim=-1)
        return target_embeddings, attention_weights, (weighted_sums, embeddings, output)

    def predict(self, ids, attention_mask=None, token_type_ids=None, position_ids=None):
        # if self.info.BERT:
        #     ids = ids.squeeze(0)
        target_embeddings, _, _ = self.forward(
            ids=ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        return target_embeddings

    def get_embeddings(self, ids, token_type_ids=None, attention_mask=None):
        if self.info.BERT:
            output = self.bert(ids, token_type_ids, attention_mask)
            output = output[0][:,  1:, :]
        else:
            output = self.embedding(ids)

        output = output.clone().detach().requires_grad_(True)
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

        if justification is not None:
            weighted_sums, attention_weights = self.attention(
                output, gold_weights=justification)
        else:
            weighted_sums, attention_weights = self.attention(output)

        target_embeddings = self.fc(weighted_sums)
        if self.info.regularize_vectors:
            # 正規化
            if self.info.affin:
                target_embeddings = target_embeddings / \
                    torch.unsqueeze(torch.norm(
                        target_embeddings, dim=-1), dim=-1)
            else:
                weighted_sums = weighted_sums / \
                    torch.unsqueeze(torch.norm(weighted_sums, dim=-1), dim=-1)
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
                    value = line.strip().split(" ")[1::]
                    pre_weight[vocab[word]] = torch.Tensor(
                        np.array(value, dtype=np.float32))
        logger.info("done")
        if not self.info.emb_train:
            logger.info(
                "The embedding layer does not get updated in the learning process.")
        return pre_weight

    def freeze_bert_pram(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_param(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_layer_param(self, layer):
        for param in self.bert.encoder.layer[layer].parameters():
            param.requires_grad = True


class BiRnnModelWithMetricFc(nn.Module):
    def __init__(self, info, metric):
        super().__init__()

        self.info = info

        if info.emb_path != None:
            pretrained_weight = self.load_embeddings()
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim).from_pretrained(
                pretrained_weight, freeze=True)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim)

        self.dropout = nn.Dropout(0.1)
        self.dropout_rnn = nn.Dropout(self.info.dropout_prob)
        self.bi_rnn = nn.LSTM(input_size=self.info.emb_dim, hidden_size=self.info.rnn_dim,
                              bidirectional=self.info.bidirectional, batch_first=True)
        self.attention = layer.Attention(self.info.rnn_dim)
        self.fc = nn.ModuleList()
        self.attention = nn.ModuleList()
        # self.dmy = nn.Parameter(torch.FloatTensor(768))
        # self.dmy.requires_grad = False

        for i, label in enumerate(self.info.ps_labels):
            self.attention.append(layer.Attention(self.info.rnn_dim))
            if metric == "arcface":
                self.fc.append(metrics.ArcMarginProduct(
                    self.info.rnn_dim, label[1] + 1))
            elif metric == "cosface":
                self.fc.append(metrics.AddMarginProduct(
                    self.info.rnn_dim, label[1] + 1))
            elif metric == "sphere":
                self.fc.append(metrics.SphereProduct(
                    self.info.rnn_dim, label[1] + 1))
        logger.info(
            f"Building {metric} model.")

    def forward(self, ids, batch_y, token_type_ids=None,
                attention_mask=None, labels=None):

        embeddings = []
        # print(attention_mask.sum(1))

        output = self.embedding(ids)

        embeddings.append(output)

        # print(output[:, attention_mask.sum(1)])

        # output, _ = self.attention(outpust)
        output = self.dropout(output)
        output = nn.utils.rnn.pack_padded_sequence(
            output,  attention_mask.sum(1), batch_first=True, enforce_sorted=False)

        output, h = self.bi_rnn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()
            # output = self.dropout_rnn(output)

        hidden_states = []
        scores = []
        attentions = []
        for i, label in enumerate(self.info.ps_labels):
            batch_y_ps = batch_y[i +
                                 1].type(torch.LongTensor).abs().to(self.info.device)
            output_attn, attention_weight = self.attention[i](output)
            output_fc = self.fc[i](output_attn, batch_y_ps)
            scores.append(output_fc)
            attentions.append(attention_weight)
            hidden_states.append(output_attn)

        return scores, attentions, (hidden_states, embeddings)

    def load_embeddings(self):
        vec_path = self.info.emb_path
        vocab = self.info.vocab
        vocab_size = len(vocab)
        pre_weight = torch.randn(vocab_size, self.info.emb_dim)

        logger.info(f"loading pretrained-vectors from {self.info.emb_path}")

        with open(vec_path, "r") as fi:
            for i, line in enumerate(fi):
                if i == 0:
                    continue
                line_list = line.strip().split(" ", 1)
                word = line_list[0]
                if word in vocab:
                    value = line.strip().split(" ")[1::]
                    pre_weight[vocab[word]] = torch.Tensor(
                        np.array(value, dtype=np.float32))
        logger.info("done")
        if not self.info.emb_train:
            logger.info(
                "The embedding layer does not get updated in the learning process.")
        return pre_weight


class BertForPartScoringWithLSTMTransductive(nn.Module):
    def __init__(self, config, t_info):
        super(BertForPartScoringWithLSTMTransductive, self).__init__()
        self.info = t_info
        # self.num_labels = config.num_labels
        if self.info.char:
            self.bert = BertModel(config).from_pretrained(
                "cl-tohoku/bert-base-japanese-char-whole-word-masking")
        else:
            self.bert = BertModel(config).from_pretrained(
                "cl-tohoku/bert-base-japanese-whole-word-masking")
        self.dropout = nn.Dropout(0.1)
        self.dropout_rnn = nn.Dropout(self.info.dropout_prob)
        self.bi_rnn = nn.LSTM(input_size=config.hidden_size, hidden_size=self.info.rnn_dim,
                              bidirectional=self.info.bidirectional, batch_first=True)
        self.attention = layer.Attention(self.info.rnn_dim)
        self.fc = nn.ModuleList()
        self.attention = nn.ModuleList()
        # self.dmy = nn.Parameter(torch.FloatTensor(768))
        # self.dmy.requires_grad = False

        for i, label in enumerate(self.info.ps_labels):
            self.attention.append(
                layer.AttentionTransductive(self.info.rnn_dim))
            self.fc.append(nn.Linear(
                self.info.rnn_dim, label[1] + 1))

    def forward(self, ids, token_type_ids=None,
                attention_mask=None, labels=None, weight=None):

        embeddings = []
        # print(attention_mask.sum(1))

        output = self.bert(ids, token_type_ids, attention_mask)

        output = output[0][:,  1:, :]

        embeddings.append(output)

        # print(output[:, attention_mask.sum(1)])

        # output, _ = self.attention(outpust)
        output = self.dropout(output)
        output = nn.utils.rnn.pack_padded_sequence(
            output,  attention_mask.sum(1) - 1, batch_first=True, enforce_sorted=False)

        output, h = self.bi_rnn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()
            # output = self.dropout_rnn(output)

        hidden_states = []
        scores = []
        attentions = []
        for i, label in enumerate(self.info.ps_labels):
            if type(weight) != type(None) and i < len(self.info.main_factors):
                output_attn, attention_weight = self.attention[i](
                    output, gold_weights=weight[i])
            else:
                output_attn, attention_weight = self.attention[i](output)
            output_fc = self.fc[i](output_attn)
            scores.append(output_fc)
            attentions.append(attention_weight)
            hidden_states.append(output_attn)

        return scores, attentions, (hidden_states, embeddings)

    def predict(self, ids, token_type_ids=None,
                attention_mask=None):
        output = self.bert(ids, token_type_ids, attention_mask)

        output = output[0][:,  1:, :]

        embeddings = []
        embeddings.append(output)

        output, h = self.bi_rnn(output)

        if self.info.bidirectional:
            output = output[:, :, :self.info.rnn_dim] + \
                output[:, :, self.info.rnn_dim:]
            output = output.contiguous()
            # output = self.dropout_rnn(output)

        hidden_states = []
        scores = []
        attentions = []
        probs = []
        for i, label in enumerate(self.info.ps_labels):
            output_attn, attention_weight = self.attention[i](output)
            output_fc = self.fc[i](output_attn)
            probs.append(output_fc.softmax(1))
            scores.append(output_fc.argmax(1))
            attentions.append(attention_weight)
            hidden_states.append(output_attn)

        return scores, probs, attentions, (hidden_states, embeddings)

    def logging_model_info(self):
        logger.info("Build Bert model for part scoring with LSTM.")

    def freeze_bert_pram(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_param(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_layer_param(self, layer):
        for param in self.bert.encoder.layer[layer].parameters():
            param.requires_grad = True
