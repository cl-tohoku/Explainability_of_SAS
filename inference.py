import json
from sklearn.neighbors import KNeighborsClassifier
import MeCab
import argparse
import torch
import model
from getinfo import TrainInfo
import pickle
from sas import input_data
from logging import getLogger
logger = getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-model")
    parser.add_argument("--train_info_path", "-info")
    parser.add_argument("--train_res", "-res", dest="res")
    parser.add_argument("--emb", "-emb")
    parser.add_argument("--in", "-i", dest="i")
    parser.add_argument("--save-path", "-save", dest="s")

    args = parser.parse_args()

    return args


def get_scores(embeddiings, train_hidden_states, train_golds, info):
    neigh = KNeighborsClassifier(n_neighbors=1)
    pred = []
    dev_dist = []
    dev_evidence = []
    ds = 0
    for i, factor in enumerate(info.main_factors):
        neigh.fit(train_hidden_states[i], train_golds[i])

        d_dist, d_pred_id = neigh.kneighbors(
            embeddiings[i].numpy(), n_neighbors=1)

        d_pred = train_golds[i + 1][d_pred_id[:, 0]]

        dev_dist.append(d_dist)
        dev_evidence.append(d_pred_id)

        # print(d_pred)
        pred.append(d_pred)

        # d_pred = np.argmax(
        #    dev_pred[i].softmax(1).cpu().detach().numpy(), axis=1)
        # t_pred = np.argmax(
        #    test_pred[i].softmax(1).cpu().detach().numpy(), axis=1)

        # logger.debug(dev_pred[i].softmax(1).cpu().detach().numpy().max(1))

        # dev_qwks[factor] = d
        # test_qwks[factor] = t
        ds += d_pred

    # for i in range(len(info.ded_factors)):
    #     num = len(info.main_factors) + i +1
    #
    #     neigh.fit(train_hidden_states[num], train_golds[num])
    #
    #     d_dist, d_pred_id = neigh.kneighbors(embeddiings[i].detach().numpy(), n_neighbors=1)
    #
    #     d_pred = train_golds[i][d_pred_id[:,0]]
    #
    #     dev_dist.append(d_dist)
    #     dev_evidence.append(d_pred_id)
    #
    #     # print(d_pred)
    #     pred.append(d_pred)
    #
    #     # d_pred = np.argmax(
    #     #    dev_pred[i].softmax(1).cpu().detach().numpy(), axis=1)
    #     # t_pred = np.argmax(
    #     #    test_pred[i].softmax(1).cpu().detach().numpy(), axis=1)
    #
    #     # logger.debug(dev_pred[i].softmax(1).cpu().detach().numpy().max(1))
    #
    #     # dev_qwks[factor] = d
    #     # test_qwks[factor] = t
    #     ds += d_pred

    return pred, dev_dist, dev_evidence


def main():
    from model import BiRnnModel
    tagger = MeCab.Tagger("-Owakati")
    args = parse_args()
    instance = json.load(open(args.i, "r"))
    info = pickle.load(open(args.train_info_path, "rb"))
    info.emb_path = args.emb
    model = BiRnnModel(info)
    model.load_state_dict(torch.load(
        args.model_path, map_location=torch.device('cpu'))["model_state_dict"])
    model.eval()
    text = instance["mecab"]
    golds = [instance["score"]]
    for factor in info.main_factors:
        golds.append(torch.LongTensor([instance[factor + "_Score"]]))
    parsed_text = tagger.parse(text)
    ids, mask = input_data.input_data_inference(
        info, parsed_text.strip().split())
    scores, attentions, (hidden_states, embeddings) = model.predict(
        ids, attention_mask=mask)
    # print(scores,attentions,hidden_states,embeddings)
    attention, hidden, ref = pickle.load(open(args.res, "rb"))
    pred, dev_dist, dev_evidence = get_scores(hidden_states, hidden, ref, info)
    pickle.dump((hidden_states, pred, golds, dev_dist[0], dev_evidence), open(
        args.s + "_" + "inference" + "_hidden_pred_trg_dist_evidence.pickle", "wb"))


if __name__ == '__main__':
    main()
