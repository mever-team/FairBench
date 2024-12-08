from fairbench.v2 import core as c
from fairbench.v2.blocks.quantities import quantities
import numpy as np


@c.measure("the average score")
def avgscore(score, sensitive=None):
    predictions = np.array(predictions)
    sensitive = np.ones_like(predictions) if sensitive is None else np.array(sensitive)
    positives = (predictions * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else positives / samples
    return c.Value(
        value, depends=[quantities.positives(positives), quantities.samples(samples)]
    )
