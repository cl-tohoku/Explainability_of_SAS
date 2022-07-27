import torch
import torch.nn
import torch.nn.functional as F


def attn_loss(input, target):
    # print(f"input:{input[0]}")
    # print(f"target:{target[0]}")
    squared_error = (input - target) ** 2
    # print("sqr")
    # print(squared_error[0])
    # print(squared_error.sum(1)[0])
    loss = squared_error.sum(1).mean()
    # print(loss)
    return loss


def modified_attn_loss(input, target, beta):
    precision_error = (F.relu(input - target) ** 2)
    recall_error = (F.relu(target - input) ** 2)
    squared_error = (beta*precision_error + (1-beta)*recall_error) * 2
    loss = squared_error.sum(1).mean()
    return loss


def cross_entropy_with_penalty(input, target, device):
    org_loss = F.cross_entropy(input, target, reduce=False)
    preds = input.argmax(1)
    penalty = (preds - target).abs().type(torch.FloatTensor).to(device)
    loss = org_loss + penalty
    return loss.mean()


def cross_entropy_with_util(input, target, device, N):
    org_loss = F.cross_entropy(input, target, reduce=False)
    preds = input.argmax(1)
    penalty = (preds - target).abs().type(torch.FloatTensor).to(device)
    loss = org_loss + penalty
    return loss.mean()


class CrossEntropyWithpenalty():
    def __init__(self):
        self.crossentropy = torch.nn.CrossEntropyLoss(reduce=False)

    def __call__(input, target):
        losses = self.crossentropy(input, target)
        preds = input.argmax(1)
        penalty = (preds - target).abs()
        loss = (losses + penalty).mean()
        return loss
