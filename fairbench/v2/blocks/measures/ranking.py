from fairbench.v2 import core as c
from fairbench.v2.blocks.quantities import quantities
import numpy as np


@c.measure("the average score")
def avgscore(scores, sensitive=None, bins=100):
    scores = np.array(scores, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    positives = (scores * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else positives / samples

    scores = scores * sensitive
    hist, bin_edges = np.histogram(
        scores[sensitive != 0], bins=bins, density=True, range=(0, 1)
    )
    bin_edges = np.concatenate([[0], bin_edges[:-1][hist != 0], [bin_edges[-1], 1]])
    hist = np.concatenate([[0], hist[hist != 0], [0]])

    curve = c.Curve(
        y=np.array((bin_edges[:-1] + bin_edges[1:]) / 2, dtype=float),
        x=np.array(hist, dtype=float),
        units="",
    )

    return c.Value(
        value,
        depends=[
            quantities.positives(positives),
            quantities.samples(samples),
            quantities.distribution(curve),
        ],
    )


@c.measure("the area under curve of the receiver operating characteristics")
def auc(scores, labels, sensitive=None):
    from fairbench.fallbacks import auc as _auc, roc_curve as _roc_curve
    import math

    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)

    median_sensitive = np.median(sensitive)
    scores = scores[sensitive >= median_sensitive]
    labels = labels[sensitive >= median_sensitive]
    fpr, tpr, _ = _roc_curve(labels, scores)
    value = _auc(fpr, tpr)

    if math.isnan(value):  # TODO: temporary solution
        return c.NotComputable(
            f"Cannot compute AUC when all instances have the same label for a sensitive attribute dimension"
        )

    assert not math.isnan(
        value
    ), f"Cannot compute AUC when all instances have the same label for a sensitive attribute dimension"

    curve = c.Curve(
        x=fpr,
        y=tpr,
        units="",
    )

    return c.Value(
        value,
        depends=[quantities.samples(sensitive.sum()), quantities.roc(curve)],
    )


@c.measure("the hit ratio of top recommendations")
def tophr(scores, labels, sensitive=None, top=3):
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)

    k = int(top)
    assert (
        0 < k <= scores.shape[0]
    ), f"There are only {scores.shape[0]} inputs but top={top} was requested for ranking analysis"

    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = np.argsort(scores)[-k:]

    value = labels[indexes].mean() if len(indexes) else 0
    true_top = labels[indexes].sum() if len(indexes) else 0
    samples = sensitive.sum()

    return c.Value(
        value,
        depends=[
            quantities.top(k),
            quantities.tp(true_top),
            quantities.samples(samples),
        ],
    )


@c.measure("the precision of top recommendations")
def toprec(scores, labels, sensitive=None, top=3):
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)

    k = int(top)
    assert (
        0 < k <= scores.shape[0]
    ), f"There are only {scores.shape[0]} inputs but top={top} was requested for ranking analysis"

    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = np.argsort(scores)[-k:]

    true_top = labels[indexes].sum()
    denom = labels.sum()
    value = 0 if denom == 0 else true_top / denom
    samples = sensitive.sum()

    return c.Value(
        value,
        depends=[
            quantities.top(k),
            quantities.tp(true_top),
            quantities.ap(denom),
            quantities.samples(samples),
        ],
    )


@c.measure("the F1 score of top recommendations")
def topf1(scores, labels, sensitive=None, top=3):
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)

    k = int(top)
    assert (
        0 < k <= scores.shape[0]
    ), f"There are only {scores.shape[0]} inputs but top={top} was requested for ranking analysis"

    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    indexes = np.argsort(scores)[-k:]

    prec = labels[indexes].mean() if len(indexes) else 0
    denom_rec = labels.sum() if len(indexes) else 0
    rec = 0 if denom_rec == 0 else labels[indexes].sum() / denom_rec
    denom = prec + rec
    value = 0 if denom == 0 else 2 * prec * rec / denom
    true_top = labels[indexes].sum()
    samples = sensitive.sum()

    return c.Value(
        value,
        depends=[
            quantities.top(k),
            quantities.tp(true_top),
            quantities.precision(prec),
            quantities.samples(samples),
        ],
    )


@c.measure("the average representation at top recommendations", unit=False)
def avgrepr(scores, sensitive=None, top=3):
    scores = np.array(scores, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)

    k = int(top)
    assert (
        0 < k <= scores.shape[0]
    ), f"There are only {scores.shape[0]} inputs but top={top} was requested for ranking analysis"

    expected = float(np.mean(sensitive))
    indexes = np.argsort(scores)[-k:]

    accum = 0
    curve = []
    for num in range(1, k + 1):
        accum += sensitive[indexes[-num]]
        curve.append(accum / num / expected)

    avg_representation = 0 if len(curve) == 0 else np.mean(curve)
    samples = sensitive.sum()

    explanation_curve = c.Curve(
        x=np.arange(1, k + 1, dtype=float),
        y=np.array(curve, dtype=float),
        units="representation",
    )

    return c.Value(
        avg_representation,
        depends=[
            quantities.top(k),
            quantities.repr(explanation_curve),
            quantities.samples(samples),
        ],
    )
