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
            tpr.append(tp / pos_count)
            fpr.append(fp / neg_count)

        if label == 1:
            tp += 1
        else:
            fp += 1

        prev_score = score
    tpr.append(tp / pos_count)
    fpr.append(fp / neg_count)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    return fpr, tpr, None


def auc(fpr, tpr):
    # Use trapezoidal rule to calculate AUC
    return np.trapz(
        tpr, fpr
    )  # TODO: this is deprecated but we need to avoid scipy calls
