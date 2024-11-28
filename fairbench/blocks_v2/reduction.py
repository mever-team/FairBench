from fairbench import core_v2 as c
import numpy as np


@c.reduction("minimum")
def minimum(values):
    values = c.transform.number(values)
    return np.min(values)


@c.reduction("minimum")
def mean(values):
    values = c.transform.number(values)
    return np.min(values)


@c.reduction("maximum difference between groups")
def maxdiff(values):
    values = c.transform.diff(values)
    return np.max(values)


@c.reduction("maximum relative difference between groups")
def maxrelative(values):
    values = c.transform.relative(values)
    return np.max(values)
