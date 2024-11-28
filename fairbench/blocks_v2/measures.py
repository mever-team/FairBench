from fairbench import core_v2 as c
import numpy as np


class quantities:
    samples = c.Descriptor("samples", "quantity")
    positives = c.Descriptor("positives", "quantity", "positive predictions")
    negatives = c.Descriptor("negatives", "quantity", "negative predictions")
    tp = c.Descriptor("tp", "quantity", "true positive predictions")
    tn = c.Descriptor("tm", "quantity", "true negative predictions")


@c.measure("positive rate")
def pr(predictions, sensitive=None):
    predictions = np.array(predictions)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    positives = (predictions * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else positives / samples
    return c.Value(
        value, depends=[quantities.positives(positives), quantities.samples(samples)]
    )


@c.measure("true positive rate")
def tpr(predictions, labels, sensitive=None):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    positives = (predictions * sensitive).sum()
    tp = (predictions * sensitive * (1 - labels)).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else tp / samples
    return c.Value(
        value,
        depends=[
            quantities.positives(positives),
            quantities.samples(samples),
            quantities.tp(tp),
        ],
    )
