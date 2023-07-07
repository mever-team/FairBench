from fairbench.forks import parallel, unit_bounded, role
from fairbench.forks.explanation import Explainable, ExplanationCurve
from eagerpy import Tensor
import numpy as np


@role("metric")
@parallel
@unit_bounded
def auc(scores: Tensor, labels: Tensor, sensitive: Tensor = None):
    import sklearn

    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels.numpy(), scores.numpy())
    return Explainable(
        sklearn.metrics.auc(fpr, tpr),
        curve=ExplanationCurve(fpr, tpr, "ROC"),
        samples=sensitive.sum(),
    )


@role("metric")
@parallel
@unit_bounded
def hr(scores: Tensor, labels: Tensor, sensitive: Tensor = None, top: int = 3):
    k = int(top.numpy())
    assert 0 < k <= scores.shape[0]
    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = scores.argsort()
    indexes = indexes[-k:]
    return Explainable(labels[indexes].mean(),
                       top=k,
                       true_top=labels[indexes].sum(),
                       samples=sensitive.sum()
                       )

@role("metric")
@parallel
@unit_bounded
def reck(scores: Tensor, labels: Tensor, sensitive: Tensor = None, top: int = 3):
    k = int(top.numpy())
    assert 0 < k <= scores.shape[0]
    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = scores.argsort()
    indexes = indexes[-k:]
    return Explainable(labels[indexes].sum()/labels.sum(),
                       top=k,
                       true_top=labels[indexes].sum(),
                       samples=sensitive.sum()
                       )

@role("metric")
@parallel
@unit_bounded
def f1k(scores: Tensor, labels: Tensor, sensitive: Tensor = None, top: int = 3):
    k = int(top.numpy())
    assert 0 < k <= scores.shape[0]
    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = scores.argsort()
    indexes = indexes[-k:]
    prec = labels[indexes].mean()
    rec = labels[indexes].sum()/labels.sum()
    return Explainable(2*prec*rec/(prec+rec),
                       top=k,
                       true_top=labels[indexes].sum(),
                       samples=sensitive.sum()
                       )


@role("metric")
@parallel
@unit_bounded
def ap(scores: Tensor, labels: Tensor, sensitive: Tensor = None, top: int = 3):
    k = int(top.numpy())
    assert 0 < k <= scores.shape[0]
    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = scores.argsort()
    curve = list()
    accum = 0
    for num in range(1, k+1):
        accum += labels[indexes[-num]].numpy()
        curve.append(accum/num*labels[indexes[-num]].numpy())
    curve = [v/k for v in curve]
    return Explainable(sum(curve),
                       top=k,
                       curve=ExplanationCurve(np.array(list(range(len(curve))), dtype=float), np.array(curve, dtype=float), "hks"),
                       samples=sensitive.sum()
                       )

