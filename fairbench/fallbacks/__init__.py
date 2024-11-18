import os

if os.environ.get("FBCONNECT_sklearn", False):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import auc, roc_curve
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError:
        print(
            "FBCONNECT_sklearn was set but sklearn was not installed, FairBench will fallback to numpy implementations"
        )
        del os.environ["FBCONNECT_sklearn"]

if not os.environ.get("FBCONNECT_sklearn", False):
    from fairbench.fallbacks.learning.logistic_regression import (
        LogisticRegression,
    )
    from fairbench.fallbacks.learning.min_max_scaler import MinMaxScaler
    from fairbench.fallbacks.learning.auc import auc, roc_curve
    from fairbench.fallbacks.read_csv import train_test_split


if os.environ.get("FBCONNECT_pandas", False):
    try:
        from pandas import read_csv, get_dummies, concat
    except ModuleNotFoundError:
        print(
            "FBCONNECT_pandas was set but pandas was not installed, FairBench will fallback to a simpler CSV reader"
        )
        del os.environ["FBCONNECT_pandas"]

if not os.environ.get("FBCONNECT_pandas", False):
    from fairbench.fallbacks.read_csv import read_csv, get_dummies, concat
