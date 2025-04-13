import os
import numpy as np

if os.environ.get("FBINTERACTIVE", False):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import auc, roc_curve
        from sklearn.model_selection import train_test_split
        from pandas import read_csv, get_dummies, concat
    except ModuleNotFoundError:
        print(
            "FBINTERACTIVE was set but either sklearn or pandas were not installed.\n"
            "FairBench will fallback to numpy implementations.\n"
            "Install `pip install --upgrade fairbench[interactive] to enable this behavior."
        )
        del os.environ["FBINTERACTIVE"]

if not os.environ.get("FBINTERACTIVE", False):
    from fairbench.fallbacks.learning.logistic_regression import (
        LogisticRegression,
    )
    from fairbench.fallbacks.learning.min_max_scaler import MinMaxScaler
    from fairbench.fallbacks.learning.auc import auc, roc_curve
    from fairbench.fallbacks.read_csv import train_test_split
    from fairbench.fallbacks.read_csv import read_csv, get_dummies, concat


def rankdata(x):
    x = np.asarray(x)
    sorter = np.argsort(x)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    x_sorted = x[sorter]
    obs = np.r_[True, x_sorted[1:] != x_sorted[:-1]]
    idx = np.flatnonzero(obs)
    idx = np.r_[idx, len(x)]
    ranks = np.empty_like(x, dtype=float)

    for start, end in zip(idx[:-1], idx[1:]):
        avg_rank = 0.5 * (start + 1 + end)
        ranks[sorter[start:end]] = avg_rank

    return ranks
