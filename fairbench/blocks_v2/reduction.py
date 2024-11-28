from fairbench import core_v2 as c
import numpy as np


@c.reduction("the minimum")
def minimum(values):
    values = c.transform.number(values)
    return np.min(values)

@c.reduction("the maximum")
def maximum(values):
    values = c.transform.number(values)
    return np.max(values)

@c.reduction("the standard deviation")
def std(values):
    values = c.transform.number(values)
    return np.std(values)

@c.reduction("the average")
def mean(values):
    values = c.transform.number(values)
    return np.mean(values)

@c.reduction("the maximum difference")
def maxdiff(values):
    values = c.transform.diff(values)
    return np.max(values)

@c.reduction("the maximum relative difference")
def maxrelative(values):
    values = c.transform.relative(values)
    return np.max(values)
