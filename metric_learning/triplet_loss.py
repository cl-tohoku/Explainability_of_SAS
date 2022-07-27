import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations


class OnlineTripletLoss(nn.Module):

    def __init__(self, margin, n_neighbors=None):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.n_neighbors = n_neighbors

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

    def get_triplets(self, embeddings, labels, n_neighbors):
        from sklearn.neighbors import KNeighborsClassifier
        labels = labels.cpu().data.numpy()
        embeddings = embeddings.cpu().detach().numpy()

        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            # 同じラベルのデータがない場合はつぎのlabelへ
            if len(label_indices) < 2:
                continue
            elif len(label_indices) <= n_neighbors:
                n_neighbors = len(label_indices) - 1

            negative_indices = np.where(np.logical_not(label_mask))[0]
            # All anchor-positive pairs
            # anchor_positives = list(combinations(label_indices, 2))
            # そのラベルのn_neignborでアンカーを作る
            # anchor_positives = list(combinations(label_indices, 2))

            # 最近某は自身のデータなのでn_neighbors+1近傍を取得する
            neigh = KNeighborsClassifier(n_neighbors=n_neighbors+1)
            neigh.fit(embeddings[label_indices], labels[label_indices])
            _, d_pred_id = neigh.kneighbors(embeddings[label_indices, :])

            # 最近某は自分自身なので削除
            d_pred_id = d_pred_id[:, 1:].tolist()
            # d_pred_idのidはknnの中でのidなのでlabels_indicesで変換
            d_pred_id = list(
                map(lambda x: list(map(lambda _x: label_indices[_x], x)), d_pred_id))
            # label_indicesと結合
            anchor_positives = np.hstack(
                (label_indices.reshape(-1, 1), d_pred_id)).tolist()
            # positive pairを作成
            anchor_positives = frozenset(
                [frozenset(positive_pair) for anchor_positive in anchor_positives for positive_pair in combinations(anchor_positive, 2)])
            # Add all negatives for all positive pairs
            temp_triplets = [[*anchor_positive, neg_ind] for anchor_positive in anchor_positives
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
            ap_distances = (embeddings[triplets[:, 0]] -
                            embeddings[triplets[:, 1]]).pow(2).sum(1)
            an_distances = (embeddings[triplets[:, 0]] -
                            embeddings[triplets[:, 2]]).pow(2).sum(1)
        except:
            return 0, 0
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class ImprovedOnlineTripletLoss(nn.Module):

    def __init__(self, margin, p_margin):
        super(ImprovedOnlineTripletLoss, self).__init__()
        self.margin = margin
        self.p_margin = p_margin

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

        triplets = self.get_all_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(ap_distances - an_distances + self.margin) + \
            F.relu(ap_distances - self.p_margin)

        return losses.mean(), len(triplets)


class TrustScoreOnlineTripletLoss(nn.Module):

    def __init__(self, margin):
        super(TrustScoreOnlineTripletLoss, self).__init__()
        self.margin = margin

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

        triplets = self.get_all_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(self.margin - (an_distances / (ap_distances +
                                                       an_distances)))

        return losses.mean(), len(triplets)


class TrustScoreOnlineTripletLoss(nn.Module):

    def __init__(self, margin):
        super(TrustScoreOnlineTripletLoss, self).__init__()
        self.margin = margin

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

        triplets = self.get_all_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(self.margin - (an_distances / (ap_distances +
                                                       an_distances)))

        return losses.mean(), len(triplets)


class OnlineRankedTripletLoss(nn.Module):

    def __init__(self, margin):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin

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

    def forward(self, embeddings, target):

        triplets = self.get_all_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class RankedTripletLoss(nn.Module):

    def __init__(self, margin, p=1, k=1):
        super(RankedTripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.k = k

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

    def forward(self, embeddings, target):

        triplets = self.get_all_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 2]]).pow(2).sum(1)
        positive_label = target[triplets[:, 0]]
        negative_label = target[triplets[:, 2]]
        penalty = torch.abs(
            positive_label - negative_label).type(torch.FloatTensor)

        losses = F.relu(ap_distances - an_distances + self.margin)
        delta = (penalty ** self.p)*self.k
        if embeddings.is_cuda:
            delta = delta.cuda()
        losses = F.relu(ap_distances - an_distances + delta * self.margin)
        return losses.mean(), len(triplets)


class NomalizedRankedTripletLoss(nn.Module):

    def __init__(self, margin, p=1, k=1):
        super(NomalizedRankedTripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.k = k

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

        triplets = self.get_all_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        try:
            ap_distances = (embeddings[triplets[:, 0]] -
                            embeddings[triplets[:, 1]]).pow(2).sum(1)
            an_distances = (embeddings[triplets[:, 0]] -
                            embeddings[triplets[:, 2]]).pow(2).sum(1)
        except:
            return 0, 0
        positive_label = target[triplets[:, 0]]
        negative_label = target[triplets[:, 2]]
        penalty = torch.abs(
            positive_label - negative_label).type(torch.FloatTensor)

        losses = F.relu(ap_distances - an_distances + self.margin)
        delta = ((penalty / max_score) ** self.p)*self.k
        if embeddings.is_cuda:
            delta = delta.cuda()
        losses = F.relu(ap_distances - an_distances + delta * self.margin)
        return losses.mean(), len(triplets)


class NormalizedDualOnlineTripletLoss(nn.Module):

    def __init__(self, margin, p=1, k=1):
        super(NormalizedDualOnlineTripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.k = k

    def get_all_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []

        for anchor_idx, anchor_label in enumerate(labels):
            for pair_idx, pair_label in enumerate(labels):
                if anchor_idx == pair_idx:
                    continue
                label_mask = (labels != pair_label)
                label_indices = np.where(label_mask)[0]
                if len(label_indices) < 2:
                    continue
                negative_indices = np.where(label_mask)[0]
                # All anchor-positive pairs

                # Add all negatives for all positive pairs
                temp_triplets = [[anchor_idx, pair_idx, neg_ind]
                                 for neg_ind in negative_indices if neg_ind != anchor_idx]
                triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))

    def forward(self, embeddings, target, max_score):

        triplets = self.get_all_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        try:
            ap_distances = (embeddings[triplets[:, 0]] -
                            embeddings[triplets[:, 1]]).pow(2).sum(1)
            an_distances = (embeddings[triplets[:, 0]] -
                            embeddings[triplets[:, 2]]).pow(2).sum(1)
        except:
            return 0, 0
        positive_label = target[triplets[:, 1]]
        negative_label = target[triplets[:, 2]]
        penalty = torch.abs(
            positive_label - negative_label).type(torch.FloatTensor)

        delta = (penalty / max_score ** self.p)*self.k
        if embeddings.is_cuda:
            delta = delta.cuda()
        losses = F.relu(ap_distances - an_distances + delta * self.margin)
        return losses.mean(), len(triplets)
