from fairbench import core_v2 as c
import numpy as np


class quantities:
    samples = c.Descriptor("samples", "quantity")
    positives = c.Descriptor("positives", "quantity", "positive predictions")
    negatives = c.Descriptor("negatives", "quantity", "negative predictions")


@c.measure("positive rate")
def pr(predictions, sensitive=None):
    predictions = np.array(predictions)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    positive = (predictions * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else positive / samples
    return c.Value(
        value, depends=[quantities.positives(positive), quantities.samples(samples)]
    )


@c.measure("true positive rate")
def tpr(predictions, sensitive=None):
    # TODO: this is a wrong placeholder for testing
    predictions = np.array(predictions)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    positive = (predictions * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else positive / samples
    return c.Value(
        value, depends=[quantities.positives(positive), quantities.samples(samples)]
    )
