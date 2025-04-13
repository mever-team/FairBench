from fairbench.v2 import core as c
from fairbench.v2.blocks.quantities import quantities
import numpy as np


@c.measure("the positive rate")
def pr(predictions, sensitive=None):
    predictions = np.array(predictions)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    positives = (predictions * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else positives / samples
    return c.Value(
        value, depends=[quantities.positives(positives), quantities.samples(samples)]
    )


@c.measure("the accuracy")
def acc(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    ap = (sensitive * labels).sum()
    an = (sensitive * (1 - labels)).sum()
    tp = (predictions * sensitive * labels).sum()
    tn = ((1 - predictions) * sensitive * (1 - labels)).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else (tp + tn) / samples
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.samples(samples),
            quantities.ap(ap),
            quantities.an(an),
            quantities.tp(tp),
            quantities.tn(tn),
        ],
    )


@c.measure("the true positive rate/recall/sensitivity/hit rate")
def tpr(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    positives = (predictions * sensitive).sum()
    ap = (labels * sensitive).sum()
    tp = (predictions * sensitive * labels).sum()
    samples = sensitive.sum()
    value = 0 if ap == 0 else tp / ap
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.samples(samples),
            quantities.positives(positives),
            quantities.ap(ap),
            quantities.tp(tp),
        ],
    )


@c.measure("the true negative rate/specificity")
def tnr(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    negatives = ((1 - predictions) * sensitive).sum()
    tn = ((1 - predictions) * sensitive * (1 - labels)).sum()
    an = ((1 - labels) * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if an == 0.0 else tn / an
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.samples(samples),
            quantities.negatives(negatives),
            quantities.an(an),
            quantities.tn(tn),
        ],
    )


@c.measure("the positive predictive value/precision")
def ppv(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    positives = (predictions * sensitive).sum()
    tp = (predictions * sensitive * labels).sum()
    ap = (labels * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if positives == 0 else tp / positives
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.samples(samples),
            quantities.positives(positives),
            quantities.ap(ap),
            quantities.tp(tp),
        ],
    )


@c.measure("the lift ratio (tpr divided by pr)", unit=False)
def lift(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)

    tp = (predictions * sensitive * labels).sum()
    ap = (labels * sensitive).sum()
    positives = (predictions * sensitive).sum()
    samples = sensitive.sum()

    tpr = 0 if ap == 0 else tp / ap
    pr = 0 if samples == 0 else positives / samples
    value = 0 if pr == 0 else tpr / pr

    return c.Value(
        value,
        depends=[
            quantities.tp(tp),
            quantities.ap(ap),
            quantities.positives(positives),
            quantities.samples(samples),
        ],
    )


@c.measure("the Matthews correlation coefficient", unit=False)
def mcc(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)

    tp = (predictions * sensitive * labels).sum()
    tn = ((1 - predictions) * sensitive * (1 - labels)).sum()
    ap = (labels * sensitive).sum()
    an = ((1 - labels) * sensitive).sum()

    fp = (predictions * sensitive * (1 - labels)).sum()
    fn = ((1 - predictions) * sensitive * labels).sum()

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    value = 0 if denominator == 0 else numerator / denominator

    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.tp(tp),
            quantities.tn(tn),
            quantities.ap(ap),
            quantities.an(an),
        ],
    )


@c.measure("the Cohen's Kappa score", unit=False)
def kappa(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)

    tp = (predictions * sensitive * labels).sum()
    tn = ((1 - predictions) * sensitive * (1 - labels)).sum()
    ap = (labels * sensitive).sum()
    an = ((1 - labels) * sensitive).sum()
    positives = (predictions * sensitive).sum()
    negatives = ((1 - predictions) * sensitive).sum()
    samples = sensitive.sum()

    po = 0 if samples == 0 else (tp + tn) / samples
    pe = (
        0 if samples == 0 else ((positives * ap + negatives * an) / (samples * samples))
    )

    value = 0 if (1 - pe) == 0 else (po - pe) / (1 - pe)

    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.tp(tp),
            quantities.tn(tn),
            quantities.ap(ap),
            quantities.an(an),
            quantities.positives(positives),
            quantities.negatives(negatives),
            quantities.samples(samples),
        ],
    )


@c.measure("the true acceptance ratio (true positives compared to all)")
def tar(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    tp = (predictions * sensitive * labels).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else tp / samples
    return c.Value(
        value,
        depends=[
            quantities.samples(samples),
            quantities.tp(tp),
        ],
    )


@c.measure("the true rejection ratio (true negatives compared to all)")
def trr(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    tn = ((1 - predictions) * sensitive * (1 - labels)).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0.0 else tn / samples
    return c.Value(
        value,
        depends=[
            quantities.samples(samples),
            quantities.tn(tn),
        ],
    )


@c.measure("the f1 score")
def f1(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)

    tp = (predictions * sensitive * labels).sum()
    tn = ((1 - predictions) * sensitive * (1 - labels)).sum()
    p = (labels * sensitive).sum()
    n = ((1 - labels) * sensitive).sum()
    fn = p - tp
    fp = n - tn

    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)

    value = (
        0
        if (precision + recall) == 0
        else 2 * (precision * recall) / (precision + recall)
    )

    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.tp(tp),
            quantities.tn(tn),
            quantities.ap(p),
            quantities.an(n),
        ],
    )
