from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations


class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.2, n_neighbors=None):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.n_neighbors = n_neighbors
        self.cos_sim = nn.CosineSimilarity()

    def get_all_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            # All anchor-positive pairs
            anchor_positives = list(combinations(label_indices, 2))

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))

    # def get_triplets(self, embeddings, labels, n_neighbors):
    #     from sklearn.neighbors import KNeighborsClassifier
    #     labels = labels.cpu().data.numpy()
    #     embeddings = embeddings.cpu().detach().numpy()

    #     triplets = []
    #     for label in set(labels):
    #         label_mask = (labels == label)
    #         label_indices = np.where(label_mask)[0]
    #         # 同じラベルのデータがない場合はつぎのlabelへ
    #         if len(label_indices) < 2:
    #             continue
    #         elif len(label_indices) <= n_neighbors:
    #             n_neighbors = len(label_indices) - 1

    #         negative_indices = np.where(np.logical_not(label_mask))[0]

    #         # 最近某は自身のデータなのでn_neighbors+1近傍を取得する
    #         neigh = KNeighborsClassifier(n_neighbors=n_neighbors+1)
    #         neigh.fit(embeddings[label_indices], labels[label_indices])
    #         _, d_pred_id = neigh.kneighbors(embeddings[label_indices, :])

    #         # 最近某は自分自身なので削除
    #         d_pred_id = d_pred_id[:, 1:].tolist()
    #         # d_pred_idのidはknnの中でのidなのでlabels_indicesで変換
    #         d_pred_id = list(
    #             map(lambda x: list(map(lambda _x: label_indices[_x], x)), d_pred_id))
    #         # label_indicesと結合
    #         anchor_positives = np.hstack(
    #             (label_indices.reshape(-1, 1), d_pred_id)).tolist()
    #         # positive pairを作成
    #         anchor_positives = frozenset(
    #             [frozenset(positive_pair) for anchor_positive in anchor_positives for positive_pair in combinations(anchor_positive, 2)])
    #         # Add all negatives for all positive pairs
    #         temp_triplets = [[*anchor_positive, neg_ind] for anchor_positive in anchor_positives
    #                          for neg_ind in negative_indices]
    #         triplets += temp_triplets

    #     return torch.LongTensor(np.array(triplets))

    def forward(self, embeddings, target, max_score):

        if self.n_neighbors is None:
            triplets = self.get_all_triplets(embeddings, target)
        else:
            triplets = self.get_triplets(embeddings, target, self.n_neighbors)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        try:
            ap_distances = (self.cos_sim(
                embeddings[triplets[:, 0]], embeddings[triplets[:, 1]]))
            an_distances = (self.cos_sim(
                embeddings[triplets[:, 0]], embeddings[triplets[:, 2]]))
        except:
            return 0, 0
        losses = F.relu(an_distances - ap_distances + self.margin)

        return losses.mean(), len(triplets)


class DotTripletLoss(nn.Module):
    def __init__(self, margin=10, n_neighbors=None):
        super(DotTripletLoss, self).__init__()
        self.margin = margin
        self.n_neighbors = n_neighbors
        self.cos_sim = nn.CosineSimilarity()

    def get_all_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            # All anchor-positive pairs
            anchor_positives = list(combinations(label_indices, 2))

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))

    def forward(self, embeddings, target, max_score):

        if self.n_neighbors is None:
            triplets = self.get_all_triplets(embeddings, target)
        else:
            triplets = self.get_triplets(embeddings, target, self.n_neighbors)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        try:
            ap_distances = (embeddings[triplets[:, 0]
                                       ] * embeddings[triplets[:, 1]]).sum(1)
            an_distances = (embeddings[triplets[:, 0]
                                       ] * embeddings[triplets[:, 2]]).sum(1)
        except:
            return 0, 0
        losses = F.relu(an_distances - ap_distances + self.margin)

        return losses.mean(), len(triplets)
