import os, wget
import pandas as pd


def read_csv(uci, *args, **kwargs):
    path = "data/" + uci.replace("\\", "/")
    if not os.path.exists(path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"+uci
        os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
        wget.download(url, path)
    return pd.read_csv(path, *args, **kwargs)


def features(df, numeric, categorical):
    dfs = [df[col] for col in numeric] + [pd.get_dummies(df[col]) for col in categorical]
    return pd.concat(dfs, axis=1).values
