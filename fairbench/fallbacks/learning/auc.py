import numpy as np


def roc_curve(labels, scores):
    desc_score_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[desc_score_indices]
    sorted_labels = labels[desc_score_indices]

    tpr = [0]
    fpr = [0]
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)

    tp = 0
    fp = 0

    prev_score = None
    for i in range(len(sorted_labels)):
        label = sorted_labels[i]
        score = sorted_scores[i]
        if prev_score is not None and score != prev_score:
            tpr.append(tp / pos_count if pos_count else 0)
            fpr.append(fp / neg_count if neg_count else 0)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    tpr.append(tp / pos_count if pos_count else 0)
    fpr.append(fp / neg_count if neg_count else 0)

    return np.array(fpr), np.array(tpr), None


def trapz(y, x):
    area = 0.0
    for i in range(1, len(x)):
        width = x[i] - x[i - 1]
        avg_height = (y[i] + y[i - 1]) / 2
        area += width * avg_height
    return area


def auc(fpr, tpr):
    return trapz(tpr, fpr)
