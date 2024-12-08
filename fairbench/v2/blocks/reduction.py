import numpy as np
from fairbench.v2 import core as c


@c.reduction("the minimum")
def min(values):
    for value in values:
        value = value.value
        if isinstance(value, c.TargetedNumber) and value.value > value.target:
            raise c.NotComputable()
    values = c.transform.number(values)
    return np.min(values)


@c.reduction("the maximum")
def max(values):
    for value in values:
        value = value.value
        if isinstance(value, c.TargetedNumber) and value.value < value.target:
            raise c.NotComputable()
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
def wmean(values, weight_by=None):
    if weight_by is None:
        from fairbench.v2 import measures

        weight_by = measures.quantities.samples
    weights = [
        value | weight_by | float for value in values
    ]  # c.reduction catches NotComputable
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


@c.reduction(
    "the maximum difference from the largest group (e.g., the whole population if included)"
)
def largestmaxdiff(values):
    values = c.transform.diff(values)
    return c.TargetedNumber(np.max(values), 0)


@c.reduction(
    "the maximum relative difference from the largest group (e.g., the whole population if included)"
)
def largestmaxrel(values):
    compared_to = c.transform.at_max_samples(values)
    values = c.transform.relative(values, compared_to=compared_to)
    return c.TargetedNumber(np.max(values), 0)
