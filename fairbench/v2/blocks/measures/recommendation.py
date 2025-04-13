from fairbench.v2 import core as c
from fairbench.v2.blocks.quantities import quantities
import numpy as np

from fairbench.v2.core import TargetedNumber


@c.measure("the average score")
def avgscore(scores, sensitive=None, bins=20):
    scores = np.array(scores, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    positives = (scores * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else positives / samples

    scores = scores * sensitive
    hist, bin_edges = np.histogram(
        scores[sensitive != 0], bins=bins, density=False, range=(0, 1)
    )
    cdf = np.cumsum(hist)
    if cdf[-1] != 0:
        cdf = cdf / cdf[-1]
    curve = c.Curve(
        x=np.array((bin_edges[:-1] + bin_edges[1:]) / 2, dtype=float),
        y=np.array(cdf, dtype=float),
        units="density",
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
    samples = sensitive.sum()
    assert samples != 0, f"Cannot compute AUC for an empty group"

    mean_sensitive = (np.max(sensitive) + np.min(sensitive)) / 2
    scores = scores[sensitive >= mean_sensitive]
    labels = labels[sensitive >= mean_sensitive]
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
        TargetedNumber(value, 1),
        depends=[quantities.samples(samples), quantities.roc(curve)],
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

    scores = scores[sensitive > 0]
    labels = labels[sensitive > 0]
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

    scores = scores[sensitive > 0]
    labels = labels[sensitive > 0]
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


@c.measure("the normalized mean reciprocal rank")
def nmrr(scores, labels, sensitive=None):
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)

    scores = scores[sensitive > 0]
    labels = labels[sensitive > 0]

    if len(labels) == 0:
        return c.Value(0.0, depends=[])

    indexes = np.argsort(scores)[::-1]
    rel = labels[indexes]

    ranks = np.where(rel == 1)[0]
    mrr_value = 0.0 if len(ranks) == 0 else 1.0 / (ranks[0] + 1)

    max_rank = len(scores)
    value = mrr_value / np.log2(max_rank + 1) if max_rank > 0 else 0.0

    true_top = rel.sum()
    samples = sensitive.sum()

    return c.Value(
        value,
        depends=[
            quantities.tp(true_top),
            quantities.samples(samples),
        ],
    )


@c.measure("the normalized entropy of the score distribution")
def nentropy(scores, sensitive=None, bins=20):
    scores = np.array(scores, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    scores = scores[sensitive > 0]

    if scores.size == 0:
        return c.Value(0.0, depends=[])

    hist, bin_edges = np.histogram(scores, bins=bins, range=(0, 1), density=False)
    total = hist.sum()
    probs = hist / total if total > 0 else np.zeros_like(hist)

    nonzero_probs = probs[probs > 0]
    raw_entropy = (
        -np.sum(nonzero_probs * np.log(nonzero_probs))
        if nonzero_probs.size > 0
        else 0.0
    )

    max_entropy = np.log2(len(hist)) if len(hist) > 0 else 1.0
    normalized_entropy = 0.0 if max_entropy == 0 else raw_entropy / max_entropy
    curve = c.Curve(
        x=np.array((bin_edges[:-1] + bin_edges[1:]) / 2, dtype=float),
        y=np.array(probs, dtype=float),
        units="probability",
    )

    samples = sensitive.sum()

    return c.Value(
        normalized_entropy,
        depends=[
            quantities.samples(samples),
            quantities.distribution(curve),
        ],
    )


@c.measure("the coverage of recommendations")
def coverage(scores, sensitive=None, top=3):
    scores = np.array(scores, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)

    k = int(top)
    assert (
        0 < k <= scores.shape[0]
    ), f"There are only {scores.shape[0]} inputs but top={top} was requested"

    scores = scores[sensitive > 0]
    indexes = np.argsort(scores)[-k:]
    unique_recommended = np.unique(indexes).size
    total_items = scores.size

    value = 0 if total_items == 0 else unique_recommended / total_items

    return c.Value(
        value,
        depends=[
            quantities.top(k),
            quantities.unique(unique_recommended),
            quantities.samples(total_items),
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

    scores = scores[sensitive > 0]
    labels = labels[sensitive > 0]
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
