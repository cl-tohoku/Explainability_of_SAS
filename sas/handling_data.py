import torch
import numpy as np
import json
import pickle
import os
from tqdm import tqdm
from logging import setLoggerClass, getLogger
from util import logger as L
from logging import setLoggerClass, getLogger
import util.html_parser.Parse_json_to_html as Parse_json_to_html
logger = getLogger(__name__)


def save_result_part_scoring_cls(info, preds, length, data_path, probs=False, attentions=None, hidden_states=None, ids=None, prefix=""):
    f = open(info.out_dir + "_" + prefix + ".json", 'w')
    d_size = len(preds[0])
    jsn = []
    for i, factor in enumerate(info.factors):
        preds[i] = preds[i].softmax(1).cpu().detach().numpy()

    if info.BERT:
        margin = 2
    else:
        margin = 0

    for num in range(d_size):
        o_data = dict()
        score = 0
        for i, factor in enumerate(info.factors):
            if factor in info.main_factors:
                score += np.argmax(preds[i][num])
            elif factor in info.ded_factors:
                score -= np.argmax(preds[i][num])

            k = factor + "_score"
            o_data[k] = int(np.argmax(preds[i][num]))
            if probs != None:
                k = factor + "_probs"
                o_data[k] = np.array2string(
                    preds[i][num], precision=3, suppress_small=True, threshold=11e3)
            if attentions != None:
                k = factor + "_attention"
                attn = attentions[i][num].cpu().detach().numpy()
                o_data[k] = np.array2string(
                    attn[:length[num] - margin], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
            if hidden_states != None and info.print_attention:
                k = factor + "_hidden"
                o_data[k] = np.array2string(
                    hidden_states[0][i][num].cpu().detach().numpy(), precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)

        o_data["score"] = int(score)

        jsn.append(o_data)
    json.dump(jsn, f, ensure_ascii=False, indent=2)
    f.close()
    # Parse_json_to_html.convert_json_to_html(data_path, info.out_dir + "_" + prefix + ".json",
    #                                         save_path=info.out_dir + "_" + prefix + ".html", char=info.char, dmy=info.dmy)


def save_result_item_scoring_cls(info, preds, length, data_path, item_num, probs=False, attentions=None, hidden_states=None, ids=None, prefix="", norms=None):
    f = open(info.out_dir + "_" + prefix + ".json", 'w')
    d_size = preds.size(0)
    jsn = []
    preds = preds.softmax(1).cpu().detach().numpy()

    if info.BERT:
        margin = 2
    else:
        margin = 0

    for num in range(d_size):
        o_data = dict()
        score = 0
        factor = info.main_factors[item_num]
        k = factor + "_score"
        o_data[k] = int(np.argmax(preds[num]))
        if probs is not None:
            k = factor + "_probs"
            o_data[k] = np.array2string(
                preds[num], precision=3, suppress_small=True, threshold=11e4)
        if attentions is not None:
            k = factor + "_attention"
            attn = attentions[num].cpu().detach().numpy()
            o_data[k] = np.array2string(
                attn[:length[num] - margin], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e4)
        if hidden_states is not None and info.print_attention:
            k = factor + "_hidden"
            o_data[k] = np.array2string(
                hidden_states[0][num].cpu().detach().numpy(), precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e4)
        if norms is not None:
            k = factor + "_norm"
            o_data[k] = np.array2string(norms[num].cpu().detach().numpy(
            ), precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e4)

        o_data["score"] = int(score)

        jsn.append(o_data)
    json.dump(jsn, f, ensure_ascii=False, indent=2)
    f.close()
    # Parse_json_to_html.convert_json_to_html(data_path, info.out_dir + "_" + prefix + ".json",
    #                                         save_path=info.out_dir + "_" + prefix + ".html", char=info.char, dmy=info.dmy)


def save_result_part_scoring_reg(info, preds, length, data_path, probs=False, attentions=None, hidden_states=None, ids=None, prefix=""):
    f = open(info.out_dir + "_" + prefix + ".json", 'w')
    d_size = len(preds[0])
    jsn = []

    if info.BERT:
        margin = 2
    else:
        margin = 0

    for num in range(d_size):
        o_data = dict()
        score = 0
        for i, factor in enumerate(info.factors):
            if factor in info.main_factors:
                score += int(preds[i][num])
            elif factor in info.ded_factors:
                score -= int(preds[i][num])

            k = factor + "_score"
            o_data[k] = int(preds[i][num])
            if probs != None:
                k = factor + "_probs"
                o_data[k] = np.array2string(
                    preds[i][num], precision=3, suppress_small=True, threshold=11e3)
            if attentions != None:
                k = factor + "_attention"
                attn = attentions[i][num].cpu().detach().numpy()
                o_data[k] = np.array2string(
                    attn[:length[num] - margin], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
            if hidden_states != None and info.print_attention:
                k = factor + "_hidden"
                o_data[k] = np.array2string(
                    hidden_states[0][i][num].cpu().detach().numpy(), precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)

        o_data["score"] = int(score)
        jsn.append(o_data)
    json.dump(jsn, f, ensure_ascii=False, indent=2)
    f.close()
    Parse_json_to_html.convert_json_to_html(data_path, info.out_dir + "_" + prefix + ".json",
                                            save_path=info.out_dir + "_" + prefix + ".html", char=info.char, dmy=info.dmy)


def save_result_part_scoring_metric(info, preds, length, data_path, probs=None, attentions=None, hidden_states=None, distance=None, evidence=None, ids=None, prefix=""):
    f = open(info.out_dir + "_" + prefix + ".json", 'w')
    d_size = len(preds[0])
    jsn = []
    for i, factor in enumerate(info.factors):
        preds[i] = preds[i]

    # clstokenwを飛ばすため
    if info.BERT:
        margin = 2
    else:
        margin = 0

    for num in range(d_size):
        o_data = dict()
        score = 0
        for i, factor in enumerate(info.factors):
            if factor in info.main_factors:
                score += preds[i][num]
            elif factor in info.ded_factors:
                score += preds[i][num]
            else:
                assert 1 == 0, f"Unknown factor {factor}"

            k = factor + "_score"
            o_data[k] = int(preds[i][num])
            if attentions != None:
                k = factor + "_attention"
                attn = attentions[i][num].cpu().detach().numpy()
                o_data[k] = np.array2string(
                    attn[:length[num] - margin], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
            if hidden_states != None and info.print_attention:
                k = factor + "_hidden"
                o_data[k] = np.array2string(
                    hidden_states[0][i][num].cpu().detach().numpy(), precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
            if distance != None:
                k = factor + "_distance"
                o_data[k] = float(distance[i][num])
            if evidence != None:
                k = factor + "_evidence"
                o_data[k] = int(evidence[i][num])

        o_data["score"] = int(score)

        jsn.append(o_data)
    json.dump(jsn, f, ensure_ascii=False, indent=2)
    f.close()
    Parse_json_to_html.convert_json_to_html_with_evidence(
        data_path, info.out_dir + "_" + prefix + ".json", train_data_path=info.train_dir, save_path=info.out_dir + "_" + prefix + ".html", char=info.char, dmy=info.dmy)


def save_result_part_scoring_metric_item_scoring(info, preds, length, data_path, item_num, probs=None, attentions=None, hidden_states=None, distance=None, evidence=None, ids=None, prefix="", norms=None, att_norm=None):
    f = open(info.out_dir + "_" + prefix + ".json", 'w')
    d_size = len(preds)
    jsn = []

    # clstokenwを飛ばすため
    if info.BERT:
        margin = 2
    else:
        margin = 0

    factor = info.main_factors[item_num]
    for num in range(d_size):
        o_data = dict()
        score = preds[num]
        k = factor + "_score"
        o_data[k] = int(preds[num])

        if attentions is not None:
            k = factor + "_attention"
            attn = attentions[num].cpu().detach().numpy()
            o_data[k] = np.array2string(
                attn[:length[num] - margin], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
        if hidden_states is not None and info.print_attention:
            k = factor + "_hidden"
            o_data[k] = np.array2string(
                hidden_states[0][num].cpu().detach().numpy(), precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
        if distance is not None:
            k = factor + "_distance"
            o_data[k] = float(distance[num])
        if evidence is not None:
            k = factor + "_evidence"
            o_data[k] = int(evidence[num])

        if norms is not None:
            k = factor + "_norm"
            o_data[k] = np.array2string(norms[num].cpu().detach().numpy(
            )[:length[num] - margin], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
        if att_norm is not None:
            k = "att_norm"
            o_data[k] = np.array2string(att_norm[num].cpu().detach().numpy(
            )[:length[num] - margin], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
        o_data["score"] = int(score)

        jsn.append(o_data)
    json.dump(jsn, f, ensure_ascii=False, indent=2)
    f.close()
    Parse_json_to_html.convert_json_to_html_with_evidence(
        data_path, info.out_dir + "_" + prefix + ".json", train_data_path=info.train_dir, save_path=info.out_dir + "_" + prefix + ".html", char=info.char, dmy=info.dmy)


def save_result_part_scoring_metric_transductive(info, preds, length, data_path, probs=None, attentions=None, hidden_states=None, distance=None, evidence=None, ids=None, prefix=""):
    f = open(info.out_dir + "_" + prefix + ".json", 'w')
    d_size = len(preds[0])
    jsn = []
    for i, factor in enumerate(info.factors):
        preds[i] = preds[i]

    # clstokenwを飛ばすため
    if info.BERT:
        margin = 2
    else:
        margin = 0

    for num in range(d_size):
        o_data = dict()
        score = 0
        for i, factor in enumerate(info.factors):
            if factor in info.main_factors:
                score += preds[i][num]
            elif factor in info.ded_factors:
                score += preds[i][num]
            else:
                assert 1 == 0, f"Unknown factor {factor}"
            k = factor + "_score"
            o_data[k] = int(preds[i][num])
            if attentions != None:
                k = factor + "_attention"
                attn = attentions[i][num].cpu().detach().numpy()
                o_data[k] = np.array2string(
                    attn[:length[num] - margin], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
            if hidden_states != None and info.print_attention:
                k = factor + "_hidden"
                o_data[k] = np.array2string(
                    hidden_states[0][i][num].cpu().detach().numpy(), precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
            if distance != None:
                k = factor + "_distance"
                o_data[k] = distance[i][num].tolist()
            if evidence != None:
                k = factor + "_evidence"

                o_data[k] = evidence[i][num].tolist()

        o_data["score"] = int(score)

        jsn.append(o_data)
    json.dump(jsn, f, ensure_ascii=False, indent=2)
    f.close()
    Parse_json_to_html.convert_json_to_html_with_evidence(
        data_path, info.out_dir + "_" + prefix + ".json", train_data_path=info.train_dir, save_path=info.out_dir + "_" + prefix + ".html", char=info.char, dmy=info.dmy, k=3)


def save_train_result_part_scoring_cls(info, model, device, eval_dataloader, prefix="train"):
    import torch
    model.eval()
    label_num = len(info.ps_labels)

    # preds=[list() for _ in range(label_num)]
    # attn=[list() for _ in range(label_num)]
    # hidden=[list() for _ in range(label_num)]
    preds = []
    attn = []
    hidden = []
    target = []
    init = True
    jsn = []
    if info.BERT:
        margin = 2
    else:
        margin = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_m, batch_a in eval_dataloader:
            batch_x = batch_x.to(device)
            batch_m = batch_m.to(device)
            batch_y = batch_y

            outputs = model(batch_x,
                            token_type_ids=None, attention_mask=batch_m)

            for i, factor in enumerate(info.factors):
                if init:
                    length = batch_m.sum(1).cpu().numpy().tolist()
                    preds.append(np.argmax(outputs[0][i].softmax(
                        1).cpu().detach().numpy(), axis=1))
                    attn.append(
                        outputs[1][i].cpu().detach().numpy().tolist())

                    hidden.append(outputs[2][0][i].cpu().detach().numpy())
                    target.append(batch_y[i].numpy())
                else:
                    length.extend(batch_m.sum(1).cpu().numpy().tolist())
                    preds[i] = np.append(preds[i], np.argmax(outputs[0][i].softmax(
                        1).cpu().detach().numpy(), axis=1), axis=0)
                    attn_list = outputs[1][i].cpu().detach().numpy().tolist()
                    attn[i].extend(attn_list)

                    hidden[i] = np.append(hidden[i], outputs[2][0]
                                          [i].cpu().detach().numpy(), axis=0)
                    target[i] = np.append(target[i], batch_y[i].numpy())

            init = False

        d_size = preds[0].shape[0]

        logger.info("Processing train data")
        for num in tqdm(range(d_size)):
            o_data = dict()
            for i, (factor, labels) in enumerate(zip(info.factors, info.ps_labels)):
                k = factor + "_score"
                o_data[k] = int(preds[i][num])
                k = factor + "_attention"
                o_data[k] = np.array2string(
                    np.array(attn[i][num][:length[num] - margin]), precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
                k = factor + "_hidden"
                o_data[k] = np.array2string(
                    hidden[i][num], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
            jsn.append(o_data)

    pickle.dump((attn, hidden, target), open(
        info.out_dir + "_" + prefix + "_attention_hidden_ref.pickle", "wb"))
    f = open(info.out_dir + "_" + prefix + ".json", 'w')
    json.dump(jsn, f, ensure_ascii=False, indent=2)
    logger.info("Done")


def save_train_result_part_scoring_cls_item_scoring(info, model, device, eval_dataloader, item_num, prefix="train"):
    import torch
    model.eval()
    label_num = len(info.ps_labels)

    # preds=[list() for _ in range(label_num)]
    # attn=[list() for _ in range(label_num)]
    # hidden=[list() for _ in range(label_num)]
    preds = []
    attn = []
    hidden = []
    target = []
    init = True
    jsn = []

    if info.BERT:
        margin = 2
    else:
        margin = 0

    i = item_num
    with torch.no_grad():
        for batch_x, batch_y, batch_m, batch_a, batch_f in eval_dataloader:
            batch_x = batch_x.to(device)
            batch_m = batch_m.to(device)
            batch_y = batch_y

            outputs = model(batch_x,
                            token_type_ids=None, attention_mask=batch_m)

            if init:
                length = batch_m.sum(1).cpu().numpy().tolist()
                if outputs[0] is not None:
                    preds = np.argmax(outputs[0].softmax(
                        1).cpu().detach().numpy(), axis=1)
                # if outputs[1] is not None:
                #     attn = outputs[1].cpu().detach().numpy().tolist()
                # if outputs[2] is not None:
                #     if outputs[2][0] is not None:
                #         hidden = outputs[2][0].cpu().detach().numpy()
                target = batch_y[i + 1].numpy()
            else:
                length.extend(batch_m.sum(1).cpu().numpy().tolist())
                if outputs[0] is not None:
                    preds = np.append(preds, np.argmax(outputs[0].softmax(
                        1).cpu().detach().numpy(), axis=1), axis=0)
                # if outputs[1] is not None:
                #     attn_list = outputs[1].cpu().detach().numpy().tolist()
                #     attn.extend(attn_list)
                # if outputs[2] is not None:
                    # if outputs[2][0] is not None:
                #     hidden = np.append(
                #         hidden, outputs[2][0].cpu().detach().numpy(), axis=0)
                target = np.append(target, batch_y[i + 1].numpy())

            init = False

        d_size = preds.shape[0]

        logger.info("Processing train data")
        factor = info.main_factors[i]
        for num in tqdm(range(d_size)):
            o_data = dict()
            if num < len(preds):
                k = factor + "_score"
                o_data[k] = int(preds[num])
            if num < len(attn):
                k = factor + "_attention"
                o_data[k] = np.array2string(
                    np.array(attn[num][:length[num] - margin]), precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
            if num < len(hidden):
                k = factor + "_hidden"
                o_data[k] = np.array2string(
                    hidden[num], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
            jsn.append(o_data)

    pickle.dump((attn, hidden, target), open(
        info.out_dir + "_" + prefix + "_attention_hidden_ref.pickle", "wb"))
    f = open(info.out_dir + "_" + prefix + ".json", 'w')
    json.dump(jsn, f, ensure_ascii=False, indent=2)
    logger.info("Done")


def save_train_result_part_scoring_metric_fc(info, model, device, eval_dataloader, prefix="train"):
    import torch
    model.eval()
    label_num = len(info.ps_labels)

    # preds=[list() for _ in range(label_num)]
    # attn=[list() for _ in range(label_num)]
    # hidden=[list() for _ in range(label_num)]
    preds = []
    attn = []
    hidden = []
    target = []
    init = True
    jsn = []
    if info.BERT:
        margin = 2
    else:
        margin = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_m, batch_a in eval_dataloader:
            batch_x = batch_x.to(device)
            batch_m = batch_m.to(device)
            batch_y = batch_y

            outputs = model(batch_x, batch_y,
                            token_type_ids=None, attention_mask=batch_m)

            for i, factor in enumerate(info.factors):
                if init:
                    length = batch_m.sum(1).cpu().numpy().tolist()
                    preds.append(np.argmax(outputs[0][i].softmax(
                        1).cpu().detach().numpy(), axis=1))
                    attn.append(
                        outputs[1][i].cpu().detach().numpy().tolist())

                    hidden.append(outputs[2][0][i].cpu().detach().numpy())
                    target.append(batch_y[i].numpy())
                else:
                    length.extend(batch_m.sum(1).cpu().numpy().tolist())
                    preds[i] = np.append(preds[i], np.argmax(outputs[0][i].softmax(
                        1).cpu().detach().numpy(), axis=1), axis=0)
                    attn_list = outputs[1][i].cpu().detach().numpy().tolist()
                    attn[i].extend(attn_list)

                    hidden[i] = np.append(hidden[i], outputs[2][0]
                                          [i].cpu().detach().numpy(), axis=0)
                    target[i] = np.append(target[i], batch_y[i].numpy())

            init = False

        d_size = preds[0].shape[0]

        logger.info("Processing train data")
        for num in tqdm(range(d_size)):
            o_data = dict()
            for i, (factor, labels) in enumerate(zip(info.factors, info.ps_labels)):
                k = factor + "_score"
                o_data[k] = int(preds[i][num])
                k = factor + "_attention"
                o_data[k] = np.array2string(
                    np.array(attn[i][num][:length[num] - margin]), precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
                k = factor + "_hidden"
                o_data[k] = np.array2string(
                    hidden[i][num], precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
            jsn.append(o_data)

    pickle.dump((attn, hidden, target), open(
        info.out_dir + "_" + prefix + "_attention_hidden_ref.pickle", "wb"))
    f = open(info.out_dir + "_" + prefix + ".json", 'w')
    json.dump(jsn, f, ensure_ascii=False, indent=2)
    logger.info("Done")


def save_result_cls(info, preds, probs=None, attentions=None, hidden_states=None, ids=None, prefix="", out_probs=False):
    f = open(info.out_dir + "_" + prefix + "_log_detail.json", 'w')
    d_size = len(preds)
    jsn = []
    hidden = hidden_states[0].cpu().detach().numpy()
    for num in range(d_size):
        o_data = dict()
        k = "score"
        o_data[k] = preds[num].item()
        if out_probs:
            k = "probs"
            o_data[k] = np.array2string(
                probs[num], precision=3, suppress_small=False, max_line_width=np.inf, threshold=11e3)
        if attentions != None:
            k = "attention"
            attn = attentions[num].cpu().detach().numpy()
            o_data[k] = np.array2string(
                attn, precision=3, suppress_small=True, max_line_width=np.inf, threshold=11e3)
        if hidden_states != None and info.print_hidden_states:
            k = "hidden"
            o_data[k] = np.array2string(hidden[num],
                                        precision=3, suppress_small=False, max_line_width=np.inf, threshold=11e3)

        jsn.append(o_data)

    json.dump(jsn, f, ensure_ascii=False, indent=2)


def save_hidden_states(info, hidden, pred, target, prefix="dev"):
    hidden_states = []
    prediction = []
    for i, factor in enumerate(info.factors):
        hidden_states.append(hidden[i].cpu().detach().numpy())
        prediction.append(
            pred[i].softmax(-1).argmax(-1).cpu().detach().numpy())
    pickle.dump((hidden_states, prediction, target), open(
        info.out_dir + "_" + prefix + "_hidden_pred_trg.pickle", "wb"))


def save_hidden_states_item(info, hidden, pred=None, target=None, attn=None, norm=None, grad=None, prefix="dev"):

    hidden_states = hidden.cpu().detach().numpy() if hidden is not None else None

    if pred is not None:
        posterior = pred.softmax(-1).max(-1)[0].cpu().detach().numpy()
        prediction = pred.softmax(-1).argmax(-1).cpu().detach().numpy()
    else:
        prediction = None
        posterior = None
    if attn is not None:
        attn = attn.cpu().detach().numpy()
    if norm is not None:
        norm = norm.cpu().detach().numpy()
    if grad is not None:
        grad = grad.cpu().detach().numpy()
    data = {"hidden_states": hidden_states, "pred": prediction,
            "gold": target, "posterior": posterior, "attentions": attn, "norms": norm, "grads": grad}
    pickle.dump(data, open(
        info.out_dir + "_" + prefix + "_result.pickle", "wb"))


def save_hidden_states_reg(info, hidden, pred, target, prefix="dev"):
    hidden_states = []
    prediction = []
    for i, factor in enumerate(info.factors):
        hidden_states.append(hidden[i].cpu().detach().numpy())
        prediction.append(
            pred[i].softmax(-1).argmax(-1).cpu().detach().numpy())
    pickle.dump((hidden_states, prediction, target), open(
        info.out_dir + "_" + prefix + "_hidden_pred_trg.pickle", "wb"))


def save_hidden_states_metric(info, hidden, pred, target, dist, ev, prefix="dev"):
    hidden_states = []
    prediction = []
    for i, factor in enumerate(info.factors):
        hidden_states.append(hidden[i].cpu().detach().numpy())
        prediction.append(pred[i])
    pickle.dump((hidden_states, prediction, target, dist, ev), open(
        info.out_dir + "_" + prefix + "_hidden_pred_trg_dist_evidence.pickle", "wb"))


def save_hidden_states_metric_item(info, outputs, pred=None, target=None, dist=None, ev=None, attn=None, norm=None, grad=None, prefix="dev"):
    if not "train" in prefix:
        if info.affin:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs[2][0]
    else:
        hidden_states = outputs
    if hidden_states is not None:
        hidden_states = hidden_states.cpu().detach().numpy()
    if attn is not None:
        attn = attn.cpu().detach().numpy()
    if norm is not None:
        norm = norm.cpu().detach().numpy()
    if grad is not None:
        grad = grad.cpu().detach().numpy()
    data = {"hidden_states": hidden_states, "pred": pred,
            "gold": target, "distance": dist, "evidence": ev, "attentions": attn, "norms": norm, "grads": grad}
    pickle.dump(data, open(info.out_dir + "_" +
                           prefix + "_result.pickle", "wb"))


def save_res(info, b_epoch, best, best_mses, test_best, test_best_mses):
    data = {"best_epoch": b_epoch, "dev_qwk": best, "dev_mse": best_mses,
            "test_qwk": test_best, "test_mse": test_best_mses}
    json.dump(data, open(info.out_dir + "_evaluation_result.json", "w"))
