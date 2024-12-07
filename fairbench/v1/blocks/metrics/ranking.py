from fairbench.v1.core import parallel, unit_bounded, role
from fairbench.v1.core import Explainable, ExplanationCurve
from fairbench.v1.core import verify
from eagerpy import Tensor
import numpy as np
from typing import Optional
import math


@role("metric")
@parallel
@unit_bounded
def avgscore(scores: Tensor, sensitive: Optional[Tensor] = None, bins: int = 100):
    bins = int(bins.numpy())
    if sensitive is None:
        sensitive = scores.ones_like()
    sum_sensitive = sensitive.sum()
    sum_positives = (scores * sensitive).sum()

    scores = scores.numpy()
    sensitive = sensitive.numpy()
    hist, bin_edges = np.histogram(
        scores[sensitive == 1], bins=bins, density=True, range=(0, 1)
    )
    bin_edges = np.concatenate([[0], bin_edges[:-1][hist != 0], [bin_edges[-1], 1]])
    hist = np.concatenate([[0], hist[hist != 0], [0]])

    return Explainable(
        0 if sum_sensitive == 0 else (sum_positives / sum_sensitive),
        samples=sum_sensitive,
        sum_scores=sum_positives,
        curve=ExplanationCurve(
            np.array((bin_edges[:-1] + bin_edges[1:]) / 2, dtype=float),
            np.array(hist, dtype=float),
            "Prob. density",
        ),
    )


@role("metric")
@parallel
@unit_bounded
def auc(scores: Tensor, labels: Tensor, sensitive: Tensor = None):
    from fairbench.fallbacks import auc as _auc, roc_curve as _roc_curve

    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    fpr, tpr, _ = _roc_curve(labels.numpy(), scores.numpy())
    value = _auc(fpr, tpr)
    verify(
        not math.isnan(value),
        f"Cannot compute AUC when all instances have the same label for branch",
    )
    return Explainable(
        value,
        curve=ExplanationCurve(fpr, tpr, "ROC"),
        samples=sensitive.sum(),
    )


@role("metric")
@parallel
@unit_bounded
def tophr(
    scores: Tensor,
    labels: Tensor,
    sensitive: Tensor = None,
    top: int = 3,
    branch: str = "branch",
):
    k = int(top.numpy())
    verify(
        0 < k <= scores.shape[0],
        f"There are only {scores.shape[0]} inputs but top={top} were requested for ranking analysis",
    )
    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = scores.argsort()
    verify(
        k <= len(indexes),
        f"There are only {len(indexes)} {branch} members but top={top} were requested for ranking analysis",
    )
    indexes = indexes[(len(indexes) - k) :]
    return Explainable(
        labels[indexes].mean(),
        top=k,
        true_top=labels[indexes].sum(),
        samples=sensitive.sum(),
    )


@role("metric")
@parallel
@unit_bounded
def toprec(scores: Tensor, labels: Tensor, sensitive: Tensor = None, top: int = 3):
    k = int(top.numpy())
    verify(
        0 < k <= scores.shape[0],
        f"There are only {scores.shape[0]} inputs but top={top} were requested for ranking analysis",
    )
    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = scores.argsort()
    verify(
        k <= len(indexes),
        f"There are only {len(indexes)} members but top={top} were requested for ranking analysis",
    )
    indexes = indexes[(len(indexes) - k) :]
    denom = labels.sum()
    return Explainable(
        0 if denom == 0 else labels[indexes].sum() / denom,
        top=k,
        true_top=labels[indexes].sum(),
        true_all=denom,
        samples=sensitive.sum(),
    )


@role("metric")
@parallel
@unit_bounded
def topf1(scores: Tensor, labels: Tensor, sensitive: Tensor = None, top: int = 3):
    k = int(top.numpy())
    verify(
        0 < k <= scores.shape[0],
        f"There are only {scores.shape[0]} inputs but top={top} were requested for ranking analysis",
    )
    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = scores.argsort()
    verify(
        k <= len(indexes),
        f"There are only {len(indexes)} members but top={top} were requested for ranking analysis",
    )
    indexes = indexes[(len(indexes) - k) :]
    prec = labels[indexes].mean()
    denom_rec = labels.sum()
    rec = 0 if denom_rec == 0 else labels[indexes].sum() / denom_rec
    denom = prec + rec
    return Explainable(
        0 if denom == 0 else 2 * prec * rec / denom,
        top=k,
        true_top=labels[indexes].sum(),
        true_all=denom_rec,
        samples=sensitive.sum(),
    )


@role("metric")
@parallel
@unit_bounded
def avghr(scores: Tensor, labels: Tensor, sensitive: Tensor = None, top: int = 3):
    k = int(top.numpy())
    verify(
        0 < k <= scores.shape[0],
        f"There are only {scores.shape[0]} inputs but top={top} were requested for ranking analysis",
    )
    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = scores.argsort()
    verify(
        k <= len(indexes),
        f"There are only {len(indexes)} members but top={top} were requested for ranking analysis",
    )
    curve = list()
    accum = 0
    for num in range(1, k + 1):
        accum += labels[indexes[len(indexes) - num]].numpy()
        curve.append(accum / num * labels[indexes[len(indexes) - num]].numpy())
    curve = [v for v in curve]
    return Explainable(
        sum(curve) / len(curve),
        top=k,
        curve=ExplanationCurve(
            np.array(list(range(len(curve))), dtype=float) + 1,
            np.array(curve, dtype=float),
            "precision",
        ),
        samples=sensitive.sum(),
    )


@role("metric")
@parallel
@unit_bounded
def avgrepr(scores: Tensor, sensitive: Tensor = None, top: int = 3):
    k = int(top.numpy())
    verify(
        0 < k <= scores.shape[0],
        f"There are only {scores.shape[0]} inputs but top={top} were requested for ranking analysis",
    )
    if sensitive is None:
        sensitive = scores.ones_like()
    expected = float(sensitive.mean().numpy())
    indexes = scores.argsort()
    verify(
        k <= len(indexes),
        f"There are only {len(indexes)} members but top={top} were requested for ranking analysis",
    )
    curve = list()
    accum = 0
    for num in range(1, k + 1):
        accum += sensitive[indexes[-num]].numpy()
        curve.append(accum / num / expected)
    curve = [v for v in curve]
    return Explainable(
        0 if len(curve) == 0 else sum(curve) / len(curve),
        top=k,
        curve=ExplanationCurve(
            np.array(list(range(len(curve))), dtype=float) + 1,
            np.array(curve, dtype=float),
            "hks",
        ),
        samples=sensitive.sum(),
    )
