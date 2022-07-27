import torch
from torch import tensor
import torch.optim as optim
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import sparsemax


class Attention(nn.Module):
    def __init__(self, input_dim, rnn_dim, op="attsum", activation="tanh", stdev=0.001):
        super(Attention, self).__init__()
        self.stdev = stdev
        self.op = op
        self.activation = activation
        self.att_W = nn.Parameter(torch.randn(input_dim, rnn_dim) * self.stdev)
        self.att_V = nn.Parameter(torch.randn(rnn_dim) * self.stdev)

    def forward(self, x, gold_weights=None):
        assert x.size(-1) == self.att_W.size(
            0), f"{x.size()}, {self.att_W.size()}"
        # (input[0],input[1],input[2]).(input[2],input[2]) →(input[0],input[1],input[2])
        # do (input[1],input[2]).(input[2],input[2]) = (input[1],input[2]) each data in minibatch
        #x_no_grad = x.detach()
        y = torch.matmul(x, self.att_W)
        if self.activation == "tanh":
            y = torch.tanh(y)

        # (input[0],input[1],input[2]).(input[2],) →(input[0],input[1])
        weights = torch.matmul(y, self.att_V)
        weights = F.softmax(weights, 1)
        #weights = self.spmax(weights)
        #weights = F.relu(weights)
        #weights = F.sigmoid(weights)

        if gold_weights is not None:
            b_max_len = x.shape[-2]
            weights = gold_weights[:, :b_max_len]
        # lstmの隠れ層のshapeに合うように重みの形を変え、並べる
        w = torch.unsqueeze(weights, dim=-1).repeat(1, 1, x.shape[2])

        # out_w = out_w.reshape(
        #     x.shape[0], x.shape[1], 1).repeat(1, 1, x.shape[2])
        # アダマール積を取る
        out = x * w

        if self.op == 'attsum':
            out = out.sum(1)
        else:
            out = out.mean(1)
        return out, weights


class AttentionTransductive(nn.Module):
    def __init__(self, rnn_dim, op="attsum", activation="tanh", stdev=0.001):
        super(AttentionTransductive, self).__init__()
        self.stdev = stdev
        self.op = op
        self.activation = activation
        self.att_W = nn.Parameter(torch.randn(rnn_dim, rnn_dim) * self.stdev)
        self.att_V = nn.Parameter(torch.randn(rnn_dim) * self.stdev)

    def forward(self, x, gold_weights=None):
        # (input[0],input[1],input[2]).(input[2],input[2]) →(input[0],input[1],input[2])
        # do (input[1],input[2]).(input[2],input[2]) = (input[1],input[2]) each data in minibatch
        #x_no_grad = x.detach()
        y = torch.matmul(x, self.att_W)
        if self.activation == "tanh":
            y = torch.tanh(y)

        # (input[0],input[1],input[2]).(input[2],) →(input[0],input[1])
        weights = torch.matmul(y, self.att_V)
        weights = F.softmax(weights, -1)

        if gold_weights is not None:
            b_max_len = x.shape[1]
            weights = gold_weights[:, :b_max_len]

        out_w = weights

        # lstmの隠れ層のshapeに合うように重みの形を変え、並べる
        out_w = out_w.reshape(
            x.shape[0], x.shape[1], 1).repeat(1, 1, x.shape[2])

        # アダマール積を取る
        out = x * out_w

        if self.op == 'attsum':
            out = out.sum(1)
        else:
            out = out.mean(1)

        return out, weights


class AttentionPreviousPaper(nn.Module):
    def __init__(self, input_dim, rnn_dim, op="attsum", activation="tanh"):
        super(AttentionPreviousPaper, self).__init__()
        self._mlp = torch.nn.Linear(input_dim, rnn_dim, bias=True)
        self.op = op
        self.activation = activation
        self._context_dot_product = torch.nn.Linear(rnn_dim, 1, bias=False)
        self.vec_dim = self._mlp.weight.size(1)

    # def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
    def forward(self, x, gold_weights=None):
        # assert mask is not None
        batch_size, sequence_length, embedding_dim = x.size()

        attn_weights = x.view(batch_size * sequence_length, embedding_dim)
        attn_weights = torch.tanh(self._mlp(attn_weights))
        attn_weights = self._context_dot_product(attn_weights)
        attn_weights = attn_weights.view(
            batch_size, -1)  # batch_size x seq_len

        attn_weights = F.softmax(attn_weights, 1)

        if gold_weights is not None:
            b_max_len = x.shape[-2]
            attn_weights = gold_weights[:, :b_max_len]

        expanded_attn_weights = attn_weights.unsqueeze(
            2).expand(batch_size, sequence_length, embedding_dim)

        out = x * expanded_attn_weights

        if self.op == 'attsum':
            out = out.sum(1)
        else:
            out = out.mean(1)

        return out, attn_weights
