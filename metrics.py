import numpy as np
from hmeasure import h_score
from sklearn.metrics import precision_recall_curve, auc

np.seterr(divide='ignore', invalid='ignore')


def compute_curves_measures(labels, scores, S, P, N):
    p, r, t = precision_recall_curve(labels, scores, pos_label=None, sample_weight=None)
    ans_auc_pr = auc(r, p)

    index = np.argsort(-scores)
    scores = -scores[index]
    labels = labels[index]
    ut = np.unique(scores, return_index=True)[1]
    ut = np.append(ut[1:] - 1, S - 1)

    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tp_rand = fp * (P / N)

    prec = tp / np.arange(1, S + 1)
    tpr = tp / P
    fpr = fp / N
    tpr_m = np.log(1 + tp) / np.log(1 + P)
    fpr_m = np.log(1 + fp) / np.log(1 + N)
    tpr_m_rand = np.log(1 + tp_rand) / np.log(1 + P)
    tpr_m_norm = (tpr_m - tpr_m_rand) / (1 - tpr_m_rand) * (1 - fpr_m) + fpr_m
    for i, item in enumerate(tpr_m_norm):
        if item != item:
            tpr_m_norm[i] = 1

    ans_prec = prec[P - 1]
    if P == 1:
        ans_auc_prec = prec[0]
    else:
        ans_auc_prec = auc(np.arange(1, P + 1), prec[0: P]) / (P - 1)

    tpr = np.append(0, tpr[ut])
    fpr = np.append(0, fpr[ut])
    fpr_m = np.append(0, fpr_m[ut])
    tpr_m_norm = np.append(0, tpr_m_norm[ut])

    ans_auc_roc = auc(fpr, tpr)
    ans_auc_mroc = auc(fpr_m, tpr_m_norm)
    return ans_prec, ans_auc_prec, ans_auc_pr, ans_auc_roc, ans_auc_mroc


def calculate_ndcg(labels, scores):
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    dcg = np.sum(sorted_labels / np.log2(np.arange(2,len(scores)+2)))

    idcg = np.sum(1 / np.log2(np.arange(2,sum(labels)+2)))

    ndcg = dcg / idcg
    return ndcg


def compute_mcc(labels, scores, P, N):
    index = np.argsort(-scores)
    labels = labels[index]
    labels = np.array(labels, dtype=np.float64)

    tp = sum(labels[0:P])
    fp = P - tp
    tn = N - fp
    fn = fp

    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc


def prediction_evaluation(labels, scores):
    S = len(scores)
    P = np.count_nonzero(labels)
    N = S - P
    prec, auc_prec, auc_pr, auc_roc, auc_mroc = compute_curves_measures(labels, scores, S, P, N)
    ndcg=calculate_ndcg(labels, scores)
    mcc = compute_mcc(labels, scores, P, N)
    h_measure = h_score(labels, scores)
    return prec, auc_prec, auc_pr, auc_roc, auc_mroc, ndcg, mcc, h_measure