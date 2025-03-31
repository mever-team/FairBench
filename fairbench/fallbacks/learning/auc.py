import numpy as np


def roc_curve(labels, scores):
    # Sort by score descending
    desc_score_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[desc_score_indices]

    # Initialize TPR and FPR arrays
    tpr = [0]
    fpr = [0]
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)

    # Track true positives and false positives
    tp = 0
    fp = 0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / pos_count)
        fpr.append(fp / neg_count)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]

    return fpr, tpr, sorted_indices


def auc(fpr, tpr):
    # Use trapezoidal rule to calculate AUC
    return np.trapz(
        tpr, fpr
    )  # TODO: this is deprecated but we need to avoid scipy calls
