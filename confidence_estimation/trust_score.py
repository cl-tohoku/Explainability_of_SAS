import argparse
import numpy as np
from scipy.spatial import KDTree

class TrustScore():
    def __init__(self,max_score,k,method=""):
        self.k = k
        self.neigh = []
        self.max_score = max_score
        self.KDTrees = [None] * (self.max_score + 1)
        # self.metric = metric
        self.method = method
        self.d_size = 0
        self.uq_labels = None
        self.inf_to_non_exist = False

    def fit(self,X,Y):
        self.uq_labels = np.unique(Y)
        assert len(self.uq_labels) != 1, "Only one label in training data."
        self.d_size = X.shape[0]
        for label in self.uq_labels:
            sp_X = X[Y == label]
            if len(sp_X) != 0:
                kdt = KDTree(sp_X)
                self.KDTrees[label] = kdt

    def get_score(self,embeddings,preds):
        uq_labels_for_preds = np.unique(preds)
        # emb_for_label = embeddings[preds == label]
        distances = np.full((embeddings.shape[0],self.max_score + 1),np.inf)
        for i,Kdt in enumerate(self.KDTrees):
            if self.KDTrees[i] is not None:
                d = self.KDTrees[i].query(embeddings)[0]
                distances[:,i] = d
            else:
                if self.inf_to_non_exist:
                #train dataに存在しない点数が入力されている時
                    distances[:,i] = np.inf
                else:
                    assert 1==2,f"Score {i} is not exist in training data."

        sorted_ind = np.argsort(distances,axis=-1)
        sorted_d = np.sort(distances,axis=-1)
        #最近傍の予測スコアを持つ点との距離
        d_to_pred = distances[range(distances.shape[0]),preds]
        # 予測スコアではない最近傍点との距離
        d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1])

        eps = np.finfo(np.float32).eps
        if self.method == "ratio":
            scores = d_to_closest_not_pred / (d_to_pred + eps)
        else:
            scores = d_to_closest_not_pred / (d_to_pred + d_to_closest_not_pred + eps)


        return scores,d_to_pred,d_to_closest_not_pred



def output_trustscore(train_pk_dir,target_pk_dir,target_dir,save,max_score):
    import pickle
    import json
    import codecs
    import pandas as pd
    train_pk = pickle.load(open(train_pk_dir,"rb"))
    target_pk = pickle.load(open(target_pk_dir,"rb"))
    target_d = json.load(codecs.open(target_dir,encoding='utf-8'))
    train_x, train_golds,_, _ = load_result(train_pk)
    target_x, target_golds,target_preds, target_posterior= load_result(target_pk)

    ts = TrustScore(max_score, k=1)
    ts.fit(train_x,train_golds)
    scores,d_to_pred,d_to_closest_not_pred = ts.get_score(target_x,target_preds)
    target_ans = load_data(target_d)
    target_pk["trustscore"] = scores
    if target_posterior is not None:
        df = pd.DataFrame({"responses":target_ans,"pred":target_preds,"gold":target_golds,"dif":np.abs(target_preds-target_golds),"trustscore":scores,"posterior":target_posterior,"d_to_pred":d_to_pred,"d_to_closest_not_pred":d_to_closest_not_pred})
    else:
        df = pd.DataFrame({"responses":target_ans,"pred":target_preds,"gold":target_golds,"dif":np.abs(target_preds-target_golds),"trustscore":scores,"d_to_pred":d_to_pred,"d_to_closest_not_pred":d_to_closest_not_pred})

    df.to_excel(save)
    pickle.dump(target_pk,open(target_pk_dir,"wb"))



def load_data(instances):
    responses = []
    for d in instances:
        responses.append(d["mecab"])

    return responses


def load_result(pk):
    x = pk["hidden_states"]
    golds = pk["gold"].cpu().detach().numpy().astype(np.int32)
    pos = pk.get("posterior",None)
    pred = pk.get("pred")
    if pred is not None:
        pred = pred.astype(np.int32)

    return x, golds, pred , pos

