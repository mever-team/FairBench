import os
import random

os.environ["FBINTERACTIVE"] = "True"  # makes fb use 3rd-party libraries (incl. pandas)
import fairbench as fb
import numpy as np
import playground.ausc.perturbation as pe
from playground.ausc.models import TorchModel
import sklearn
from matplotlib import pyplot as plt


def run_all(dataset, fair, ylabel):
    splits = {"predict": "data", "seed": 42, "test_size": 0.2}

    if "compas" in dataset:
        x_train, y_train, x, y, train, test = fb.bench.tabular.compas(**splits)
        if "sex" in dataset:
            to_sensitive = lambda data: (data["sex"] == "Female").to_numpy()
        else:
            to_sensitive = lambda data: (data["race"] == "African-American").to_numpy()
        y_train = 1 - y_train
        y = 1 - y

    if "bank" in dataset:
        x_train, y_train, x, y, train, test = fb.bench.tabular.bank(**splits)
        to_sensitive = lambda data: ((data["age"] < 25) | (data["age"] > 60)).to_numpy()

    if "adult" in dataset:
        splits = {"predict": "data"}
        x_train, y_train, x, y, train, test = fb.bench.tabular.adult(**splits)
        if "sex" in dataset:
            to_sensitive = lambda data: (data[9] == "Female").to_numpy()
        else:
            to_sensitive = lambda data: (data[8] == "Black").to_numpy()

    ###### EXPERIMENTATION
    sensitive = to_sensitive(train)
    if "lr" in dataset:
        classifier = sklearn.linear_model.LogisticRegression(max_iter=1000)
        # classifier = sklearn.linear_model.SGDClassifier()
    else:
        classifier = TorchModel(x_train.shape[1], sensitive=sensitive)
    classifier.fit(x_train, y_train)
    yhat_train = classifier.predict(x_train)
    pe.run(
        yhat_train,
        y_train,
        sensitive=sensitive,
        color="b",
        label="Steepest collapse",
        n=1,
        fairness=fair,
        step=200 if "adult" in dataset else 20,
    )

    accs = list()
    fairs = list()
    sensitive = to_sensitive(test)
    # candidates = list(range(0, len(sensitive), 1))
    candidates = [i for i in range(len(sensitive)) if sensitive[i]]
    # candidates = list(range(int(sensitive.sum())))
    print("----------------------")
    for t in range(0, len(candidates) + 1):
        y_test = y
        x_test = x
        yhat = classifier.predict(x_test)
        for i in random.sample(candidates, t):
            yhat[i] = 1 - yhat[i]
        accs.append(pe.accuracy(yhat, y_test))
        fairs.append(fair(yhat, y_test, sensitive))

    plt.scatter(
        accs,
        fairs,
        color="g",
        marker=".",
        label="Testing simulation: minority perturbations",
    )

    accs = list()
    fairs = list()
    candidates = [i for i in range(len(sensitive)) if sensitive[i] and not y[i]]
    for t in range(0, len(candidates) + 1):
        y_test = y
        x_test = x
        yhat = classifier.predict(x_test)
        for i in random.sample(candidates, t):
            y_test[i] = 1 - y_test[i]
        accs.append(pe.accuracy(yhat, y_test))
        fairs.append(fair(yhat, y_test, sensitive))

    plt.scatter(
        accs,
        fairs,
        color="y",
        marker="h",
        label="Testing simulation: increased minority benefits",
    )

    sensitive = to_sensitive(train)
    plt.scatter(
        [pe.accuracy(yhat_train, y_train)],
        [fair(yhat_train, y_train, sensitive=sensitive)],
        color="r",
        marker="x",
        label="Training data",
    )

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("accuracy")
    plt.ylabel(ylabel)
    plt.legend()


fairs = [
    {"fair": pe.prule, "ylabel": "collapsed prule"},
    {"fair": pe.eq_odds, "ylabel": "collapsed eo"},
    {"fair": pe.pred_parity, "ylabel": "collapsed pp"},
]

i = 1


for setting in fairs:
    plt.subplot(5, 3, i)
    run_all("adult lr race", **setting)
    plt.title("LR - Adult (race)")
    i += 1

for setting in fairs:
    plt.subplot(5, 3, i)
    run_all("adult lr sex", **setting)
    plt.title("LR - Adult (sex)")
    i += 1

for setting in fairs:
    plt.subplot(5, 3, i)
    run_all("bank lr", **setting)
    plt.title("LR - Bank (age)")
    i += 1

for setting in fairs:
    plt.subplot(5, 3, i)
    run_all("compas lr race", **setting)
    plt.title("LR - Compas (race)")
    i += 1

for setting in fairs:
    plt.subplot(5, 3, i)
    run_all("compas lr sex", **setting)
    plt.title("LR - Compas (sex)")
    i += 1


plt.show()
