import numpy as np
from fairbench.v2 import core as c
from fairbench.v2.core import NotComputable


@c.reduction("the minimum")
def min(values):
    min_target = None
    for value in values:
        value = value.value
        if isinstance(value, c.TargetedNumber):
            if value.value > value.target:
                raise c.NotComputable()
            if min_target is None or float(value.target) > min_target:
                min_target = float(value.target)
    values = c.transform.number(values)
    ret = np.min(values) if len(values) else 0
    return ret if min_target is None else c.TargetedNumber(ret, min_target)


@c.reduction("the maximum")
def max(values):
    max_target = None
    for value in values:
        value = value.value
        if isinstance(value, c.TargetedNumber):
            if value.value < value.target:
                raise c.NotComputable()
            elif max_target is None or float(value.target) < max_target:
                max_target = float(value.target)
    values = c.transform.number(values)
    ret = np.max(values) if len(values) else 0
    return ret if max_target is None else c.TargetedNumber(ret, max_target)


@c.reduction("the maximum deviation from the ideal value")
def maxerror(values):
    for value in values:
        value = value.value
        if not isinstance(value, c.TargetedNumber):
            raise c.NotComputable()
    return c.TargetedNumber(
        np.max([abs(value.value.value - value.value.target) for value in values]), 0
    )


@c.reduction("the standard deviation x2")
def stdx2(values):
    values = c.transform.number(values)
    return c.TargetedNumber(2 * np.std(values), 0)


@c.reduction("the gini coefficient")
def gini(values):
    values = c.transform.number(values)
    n = len(values)
    if n == 0:
        return c.TargetedNumber(0, 0)
    mean = sum(values) / n
    gini_sum = sum(abs(values[i] - values[j]) for i in range(n) for j in range(n))
    gini_coefficient = 0 if mean == 0 else gini_sum / (2 * n * n * mean)
    return c.TargetedNumber(gini_coefficient, 0)


@c.reduction("the average")
def mean(values):
    values = c.transform.number(values)
    return np.mean(values) if len(values) else 0


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


@c.reduction("the maximum area between curves")
def maxbarea(values):
    values = c.transform.curve_diff(values)
    return c.TargetedNumber(np.max(values), 0)


@c.reduction("the maximum relative difference")
def maxrel(values):
    values = c.transform.relative(values)
    return c.TargetedNumber(np.max(values), 0)


@c.reduction(
    "the maximum difference from the largest group (the whole population if included)"
)
def largestmaxdiff(values):
    values = c.transform.diff(values)
    return c.TargetedNumber(np.max(values), 0)


@c.reduction(
    "the maximum relative difference from the largest group (the whole population if included)"
)
def largestmaxrel(values):
    compared_to = c.transform.at_max_samples(values)
    values = c.transform.relative(values, compared_to=compared_to)
    return c.TargetedNumber(np.max(values), 0)


@c.reduction(
    "the maximum area between curves and the curve of the largest group (the whole population if included)"
)
def largestmaxbarea(values):
    compared_to = c.transform.at_max_samples(values)
    values = c.transform.curve_diff(values, compared_to=compared_to)
    return c.TargetedNumber(np.max(values), 0)
