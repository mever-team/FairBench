import os
import pandas as pd
import zipfile
import re
from fairbench.bench import wget


def _extract_nested_zip(file, folder):
    print(file, folder)
    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(file, "r") as zfile:
        zfile.extractall(path=folder)
    os.remove(file)
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".zip"):
                _extract_nested_zip(
                    os.path.join(root, filename), os.path.join(root, filename[:-4])
                )


def read_csv(url, *args, **kwargs):
    url = url.replace("\\", "/")
    if ".zip/" in url:
        url, path = url.split(".zip/", 1)
        extract_to = "data/"
        if "/" not in path:
            extract_to += url.split("/")[-1]
            path = os.path.join(url.split("/")[-1], path)
        path = os.path.join("data", path)
        url += ".zip"
        temp = "data/" + url.split("/")[-1]
        if not os.path.exists(path):
            os.makedirs(os.path.join(*path.split("/")[:-1]), exist_ok=True)
            wget.download(url, temp)
            _extract_nested_zip(temp, extract_to)
    else:
        shortened = "/".join(url.split("/")[-4:])
        path = "data/" + shortened
        if not os.path.exists(path):
            os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
            wget.download(url, path)
    return pd.read_csv(path, *args, **kwargs)


def features(df, numeric, categorical):
    dfs = [df[col] for col in numeric] + [
        pd.get_dummies(df[col]) for col in categorical
    ]
    return pd.concat(dfs, axis=1).values
