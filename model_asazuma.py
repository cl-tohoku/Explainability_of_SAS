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


class LSTMAttention(torch.nn.Module):

    def __init__(self, info, config=None):
        super().__init__()
        self.info = info
        self.metric_name = "CrossEntropy"
        self.hidden_size = self.info.rnn_dim
        if info.emb_path != None:
            pretrained_weight = self.load_embeddings()
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim).from_pretrained(
                pretrained_weight, freeze=not info.update_embed)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.info.vocab_size, embedding_dim=self.info.emb_dim)

        # bilstmではないこととbatchfirstではないことと層が3層あることが舟山と違う
        # self.lstm = nn.LSTM(input_size=self.info.emb_dim, hidden_size=self.info.rnn_dim,  num_layers=3, dropout=0.2)
        self.lstm = nn.LSTM(input_size=self.info.emb_dim, hidden_size=self.info.rnn_dim,
            bidirectional=self.info.bidirectional, num_layers=info.lstm_num,dropout=0.2)

        self._predict = nn.Linear(info.rnn_dim, (self.info.item_max_score + 1))

        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.device = info.device

    def embedding_layer(self, input_sequences):
        embeddings = self.embedding(input_sequences)
        embeddings = embeddings.permute(1, 0, 2)
        return embeddings

    def lstm_layer(self, embeddings, batch_size):
        h_0 = torch.zeros(self.info.lstm_num*(self.info.bidirectional+1), batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.info.lstm_num*(self.info.bidirectional+1), batch_size, self.hidden_size).to(self.device)
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(embeddings, (h_0, c_0))
        lstm_output = lstm_output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)
        if self.info.bidirectional:
            lstm_output = lstm_output[:, :, :self.info.rnn_dim] + \
                lstm_output[:, :, self.info.rnn_dim:]
            lstm_output = lstm_output.contiguous()
        return lstm_output, final_hidden_state[-1].unsqueeze(0)

    def attention_layer(self, lstm_output, final_state, batch_size):
        hidden = torch.reshape(final_state, (batch_size, self.hidden_size))  # (2, B, H) -> (B, 2*H)
        orig_attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)  # (B, fix)
        attn_weights = F.softmax(orig_attn_weights, 1)  # (B, fix)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        attn_output = self.tanh(new_hidden_state)
        return attn_output, attn_weights # (B, Hidden)

    def prediction_layer(self, attn_output):
        prediction = self._predict(attn_output)
        return prediction

    # def forward(self, input_sentences, batch_size=None, prediction_only=True):
    def forward(self, ids, token_type_ids=None, position_ids=None,
                attention_mask=None, labels=None, justification=None):
        # asazuma実装の場合はpaddingする必要がない？→padにもアテンションするようになってる

        #################
        batch_size = ids.shape[0]
        output = self.embedding_layer(ids)
        output, hidden = self.lstm_layer(output, batch_size)
        norm = torch.norm(output, dim=-1).detach()
        output, attn_weights = self.attention_layer(output, hidden, batch_size)
        output = self.prediction_layer(output)


        return output, attn_weights, (None, None, norm)
    
    def predict(self, ids, attention_mask=None, token_type_ids=None, position_ids=None):
        target_embeddings, _, _ = self.forward(
            ids=ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        return target_embeddings

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

    def print_model_info(self):
        logger.info(f"Train model with {self.metric_name}")