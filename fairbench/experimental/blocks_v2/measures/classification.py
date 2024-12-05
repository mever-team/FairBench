from fairbench.experimental import core_v2 as c
from fairbench.experimental.blocks_v2.quantities import quantities
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


@c.measure("the true positive rate")
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
        value,
        depends=[
            quantities.samples(samples),
            quantities.positives(positives),
            quantities.ap(ap),
            quantities.tp(tp),
        ],
    )


@c.measure("the true negative rate")
def tnr(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    positives = (predictions * sensitive).sum()
    tn = ((1 - predictions) * sensitive * (1 - labels)).sum()
    an = ((1 - labels) * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if an == 0.0 else tn / an
    return c.Value(
        value,
        depends=[
            quantities.samples(samples),
            quantities.positives(positives),
            quantities.an(an),
            quantities.tn(tn),
        ],
    )


@c.measure("the true acceptance rate")
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


@c.measure("the true rejection rate")
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
