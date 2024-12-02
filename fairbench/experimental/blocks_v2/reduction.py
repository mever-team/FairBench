from fairbench.experimental import core_v2 as c
import numpy as np


@c.reduction("the minimum")
def min(values):
    values = c.transform.number(values)
    return np.min(values)


@c.reduction("the maximum")
def max(values):
    values = c.transform.number(values)
    return np.max(values)


@c.reduction("the standard deviation")
def std(values):
    values = c.transform.number(values)
    return c.TargetedNumber(np.std(values), 0)


@c.reduction("the average")
def mean(values):
    values = c.transform.number(values)
    return np.mean(values)


@c.reduction("the weighted average")
def wmean(values):
    from fairbench.experimental.v2 import measures

    weights = [value | measures.quantities.samples | float for value in values]
    values = c.transform.number(values)
    values = np.array(values)
    weights = np.array(weights)
    weights_sum = weights.sum()
    if weights_sum == 0:
        return values
    return np.sum(values * weights / weights_sum)


@c.reduction("the maximum difference")
def maxdiff(values):
    values = c.transform.diff(values)
    return c.TargetedNumber(np.max(values), 0)


@c.reduction("the maximum relative difference")
def maxrel(values):
    values = c.transform.relative(values)
    return c.TargetedNumber(np.max(values), 0)
