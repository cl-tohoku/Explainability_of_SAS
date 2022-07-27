from sklearn.neighbors import NearestNeighbors
from torch import Tensor, neg
import numpy as np


class KnnModel():
    def __init__(self, train_embeddings, train_labels):
        if type(train_labels) == Tensor:
            train_labels = train_labels.detach().to("cpu").numpy()
        self.train_labels = np.array(train_labels, dtype=np.int)
        if type(train_embeddings) == Tensor:
            train_embeddings = train_embeddings.detach().to("cpu").numpy()

        self.train_embeddings = train_embeddings

        self.pos_embedding = None
        self.neg_embedding = None
        # self.initial_pos_dist = None
        # self.initial_neg_dist = None

        self.index2label = sorted(list(set(train_labels)))
        self.label2index = dict((k, i) for i, k in enumerate(self.index2label))
        self.same_localindex2globalindex = [[]
                                            for _ in range(len(self.index2label))]
        for index, label in enumerate(train_labels):
            self.same_localindex2globalindex[self.label2index[label]].append(
                index)

        self.different_localindex2globalindex = [[]
                                                 for _ in range(len(self.index2label))]

        for index, label in enumerate(train_labels):
            for _label in self.index2label:
                if label == _label:
                    continue
                self.different_localindex2globalindex[self.label2index[_label]].append(
                    index)

        self.neign = NearestNeighbors(n_neighbors=1)
        self.neign.fit(train_embeddings)

        self.same_label_neign = [NearestNeighbors(
            n_neighbors=1) for _ in range(len(self.index2label))]
        self.different_label_neign = [NearestNeighbors(
            n_neighbors=1) for _ in range(len(self.index2label))]

        for index, label in enumerate(self.index2label):
            same_label_index = train_labels == label
            different_label_index = train_labels != label

            self.same_label_neign[index].fit(
                train_embeddings[same_label_index])
            self.different_label_neign[index].fit(
                train_embeddings[different_label_index])

    def get_same_label_embeddings(self, embeddings, label, k=1):
        _, neigh_ind = self.same_label_neign[self.label2index[label]].kneighbors(
            embeddings)

        return np.array([self.train_embeddings[self.same_localindex2globalindex[self.label2index[label]][ind]] for ind in neigh_ind.reshape(-1)])
        # neigh_dist, neigh_ind = self.same_label_neign[self.label2index[label]].kneighbors(
        #     embeddings)
        # # print(len(self.same_localindex2globalindex))
        # return self.train_embeddings[self.same_localindex2globalindex[self.label2index[label]][neigh_ind[0][0]]]

    def get_different_label_embeddings(self, embeddings, label, k=1):
        _, neigh_ind = self.different_label_neign[self.label2index[label]].kneighbors(
            embeddings)
        return np.array([self.train_embeddings[self.different_localindex2globalindex[self.label2index[label]][ind]] for ind in neigh_ind.reshape(-1)])
        # neigh_dist, neigh_ind = self.different_label_neign[self.label2index[label]].kneighbors(
        #     embeddings)
        # return self.train_embeddings[self.different_localindex2globalindex[self.label2index[label]][neigh_ind[0][0]]]

    def predict(self, embeddings, k=1, is_trust_score=False):
        neigh_dist, neigh_ind = self.neign.kneighbors(embeddings)
        neigh_dist = neigh_dist.reshape(-1)
        preds = [self.train_labels[ni] for ni in neigh_ind.reshape(-1)]
        if is_trust_score:
            # diff_neigh_dist, _ = zip(*[self.different_label_neign[self.label2index[self.train_labels[ni]]].kneighbors(
            #     embeddings) for ni in neigh_ind.reshape(-1)])
            # diff_neigh_dist = diff_neigh_dist[0].reshape(-1)
            # trust_score = self.calc_trust_score(neigh_dist, diff_neigh_dist)
            pos_dist, neg_dist = self.get_pos_neg_dist(embeddings)
            trust_score = neg_dist / (pos_dist + neg_dist)
            return preds, trust_score
        else:
            return preds

    def reset_pos_neg_instance(self):
        self.pos_embedding = None
        self.neg_embedding = None
        # self.initial_pos_dist = None
        # self.initial_neg_dist = None

    def set_pos_neg_instance(self, embeddings):
        # labelを予測
        # _, labels = self.neign.kneighbors(embeddings)
        # labels = labels.reshape(-1).tolist()
        pos_neigh_dist, neigh_ind = self.neign.kneighbors(embeddings)
        pos_neigh_dist = pos_neigh_dist.reshape(-1)
        labels = [self.train_labels[ni] for ni in neigh_ind.reshape(-1)]

        # まずposinegaの最近傍事例のindexを取得→indexを全体のindexに変更する→embeddingを取得→selfに格納する
        pos_embeddings = [self.get_same_label_embeddings(
            embeddings[i].reshape(1, -1), labels[i]) for i in range(len(labels))]
        neg_embeddings = [self.get_different_label_embeddings(
            embeddings[i].reshape(1, -1), labels[i]) for i in range(len(labels))]
        self.pos_embedding = np.array(pos_embeddings)
        self.neg_embedding = np.array(neg_embeddings)

        # neg_neigh_dist = self.calc_euclid_dist(embeddings, self.neg_embedding)
        # self.initial_pos_dist = pos_neigh_dist
        # self.initial_neg_dist = neg_neigh_dist

        # print("pos_embedding", self.pos_embedding)
        # print("initial_probs", self.initial_probs)

    def get_pos_neg_dist(self, embedding):
        pos_dist = self.calc_euclid_dist(self.pos_embedding, embedding)
        neg_dist = self.calc_euclid_dist(self.neg_embedding, embedding)
        return pos_dist, neg_dist

    # def get_probs(self, embeddings):
    #     pos_dist = self.calc_euclid_dist(self.pos_embedding, embeddings)
    #     neg_dist = self.calc_euclid_dist(self.neg_embedding, embeddings)
    #     probs =
    #     print("after probs", probs)
    #     return probs

    @staticmethod
    def calc_trust_score(pos_dist, neg_dist):
        return neg_dist / (pos_dist + neg_dist)

    @staticmethod
    def calc_euclid_dist(a, b):
        return np.linalg.norm(a - b, axis=-1)
