import argparse
import argparse
import pandas as pd
import numpy as np
import json
import pickle
import getinfo
import os.path as path

def get_item_name(name):
    str = path.basename(name)[:13]
    if "A" in str:
        return "A"
    elif "B" in str:
        return "B"
    elif "C" in str:
        return "C"
    elif "D" in str:
        return  "D"
    else:
        print(f"Parse error: item name, {str}")
        exit()

def mean_squared_error(rater_a,rater_b,max_score = None):
    if max_score is not None:
        rater_a = rater_a / max_score
        rater_b = rater_b / max_score

    rmse = np.sqrt(np.square(rater_a - rater_b)).mean()

    return rmse


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the count   s of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.

    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    rater_a = np.where(rater_a < 0, 0, rater_a)
    rater_b = np.where(rater_b < 0, 0, rater_b)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pk_path","-pk",dest="pk")
    parser.add_argument("-prefix")



    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    trustscores,posterior,preds,golds = load_pickle_data(args.pk)
    # item_max_score = ranges[args.item][args.prompt][1]
    # csrate_per_prop = calc_csrate_per_prop(test_df,item_max_score)
    prompt = getinfo.prompt_check(args.pk)
    item = get_item_name(args.pk)
    ranges = getinfo.get_ps_ranges()
    low,high = ranges[item][prompt]
    print(high)
    qwk_per_prop_trust = calc_mse_per_prop(trustscores,preds,golds,high)
    if posterior is not None:
        qwk_per_prop_pos = calc_mse_per_prop(posterior,preds,golds,high)
    else:
        qwk_per_prop_pos = None
    res = {"trustscore":qwk_per_prop_trust,"posterior":qwk_per_prop_pos}
    save_to_json(res,args.prefix)


def main2():
    args = parse_args()
    dev_df = pd.read_excel(args.dev)
    test_df = pd.read_excel(args.test)
    ranges = part_scoring_info.get_ps_ranges()
    item_max_score = ranges[args.item][args.prompt][1]
    tau = determine_tau(dev_df,item_max_score,l = args.l)
    num_cses,csrate,reliable_num,prop,thres = calc_csrate_from_tau(test_df,item_max_score,tau,l=args.l)
    res = {"#CSEs":int(num_cses),"CSRate":float(csrate * 100),"#prop":int(reliable_num),"Prop":float(prop * 100),"tau":tau,"threshold":thres}
    save_to_json(res,args.prefix)

def determine_tau(dev_df,max_score,l= 0.1,confidence="trust_score"):
    dev_confidences = dev_df[confidence].values
    difs = dev_df["dif"].values
    sorted_indexes = (-1.0 * dev_confidences).argsort()
    sorted_difs = difs[sorted_indexes]
    threshold = max_score * l
    judge_cses = np.where(sorted_difs >= threshold, 1, 0)
    border_num = check_first_cses(judge_cses)
    if border_num == 0:
        tau = 1.0
    elif border_num == (len(judge_cses) - 1):
        tau = dev_confidences[sorted_indexes[border_num]]
    else:
        tau = dev_confidences[sorted_indexes[border_num - 1]]

    return tau


def calc_csrate_from_tau(test_df,max_score,tau,l= 0.1,confidence="trust_score"):
    test_confidences = test_df[confidence].values
    difs = test_df["dif"].values
    reliable_idx = np.where(test_confidences >= tau)[0]
    if len(reliable_idx) != 0:
        reliable_predictions_difs = difs[reliable_idx]
        threshold = max_score * l
        cses = np.where(reliable_predictions_difs >= threshold,1,0)
        num_cses = cses.sum()
        reliable_num = len(reliable_predictions_difs)
        prop = reliable_num / len(test_confidences)
        csrate = num_cses / len(reliable_predictions_difs)
    else:
        num_cses = 0
        csrate = 0
        reliable_num = 0
        prop = 0
        threshold = max_score * l

    return num_cses,csrate,reliable_num,prop,threshold

def calc_qwk_per_prop(confidence_scores,preds,golds,props=[30,40,50,60,70,80,90,100]):
    sorted_indexes = (-1.0 * confidence_scores).argsort()
    size = confidence_scores.shape[0]
    prop_qwk_pairs = []
    for p in props:
        border_idx = int(size * (p* 0.01)) - 1
        tau = confidence_scores[sorted_indexes[border_idx]]
        target_idx = np.where(confidence_scores >= tau)[0]
        target_pred = preds[target_idx]
        target_golds = golds[target_idx]
        qwk = quadratic_weighted_kappa(target_pred,target_golds)
        prop_qwk_pairs.append((p,qwk))
        print((p,qwk))

    return prop_qwk_pairs

def calc_mse_per_prop(confidence_scores,preds,golds,max_score,props=[30,40,50,60,70,80,90,100]):
    sorted_indexes = (-1.0 * confidence_scores).argsort()
    size = confidence_scores.shape[0]
    prop_qwk_pairs = []
    for p in props:
        border_idx = int(size * (p* 0.01)) - 1
        tau = confidence_scores[sorted_indexes[border_idx]]
        target_idx = np.where(confidence_scores >= tau)[0]
        target_pred = preds[target_idx]
        target_golds = golds[target_idx]
        qwk = mean_squared_error(target_pred,target_golds,max_score=max_score)
        prop_qwk_pairs.append((p,qwk))
        print((p,qwk))

    return prop_qwk_pairs

def calc_csrate_per_prop(test_df,max_score,props=[10,20,30,40,50,60,70,80,90,100],l= 0.1,confidence="trust_score"):
    test_confidences = test_df[confidence].values
    difs = test_df["dif"].values
    predictions = test_df["Prediction"].values
    golds = test_df["Reference"].values

    size = len(difs)
    sorted_indexes = (-1.0 * test_confidences).argsort()
    prop_csrate_pairs = []
    for p in props:
        border_idx = int(size * (p* 0.01)) - 1
        tau = test_confidences[sorted_indexes[border_idx]]
        target_idx = np.where(test_confidences >= tau)[0]
        target_difs = difs[target_idx]
        threshold = max_score * l
        cses = np.where(target_difs >= threshold,1,0)
        num_cses = cses.sum()
        reliable_num = len(target_difs)
        prop = reliable_num / len(target_difs)
        csrate = num_cses / len(target_difs)
        prop_csrate_pairs.append((p,csrate))
        print((p,csrate))

    return prop_csrate_pairs

def check_first_cses(instances):
    for i, val in enumerate(instances):
        if val == 1:
            break
    return i

def save_to_json(dic,prefix):
    fpw = open(prefix + ".json","w")
    json.dump(dic,fpw,indent=1,ensure_ascii=False)

def load_pickle_data(pk_path):
    data = pickle.load(open(pk_path,"rb"))
    preds = data["pred"]
    golds = data["gold"].cpu().detach().numpy()
    dif = np.abs(preds - golds)
    posterior = data.get("posterior",None)
    return data["trustscore"],posterior,preds,golds

if __name__ == '__main__':
    main()


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the count   s of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings







def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.

    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    rater_a = np.where(rater_a < 0, 0, rater_a)
    rater_b = np.where(rater_b < 0, 0, rater_b)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator



