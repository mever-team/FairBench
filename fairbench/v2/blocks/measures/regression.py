from fairbench.v2 import core as c
from fairbench.v2.blocks.quantities import quantities
import numpy as np


@c.measure("mean absolute error")
def mabs(scores, targets, sensitive=None):
    scores = np.array(scores, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    error = (np.abs(scores - targets) * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0.0 else error / samples
    return c.Value(
        c.TargetedNumber(value, 0),
        depends=[
            quantities.samples(samples),
        ],
    )


@c.measure("root mean square error")
def rmse(scores, targets, sensitive=None):
    scores = np.array(scores, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    error = ((scores - targets) ** 2 * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0.0 else error / samples
    return c.Value(
        c.TargetedNumber(value**0.5, 0),
        depends=[
            quantities.samples(samples),
        ],
    )


@c.measure("mean square error")
def mse(scores, targets, sensitive=None):
    scores = np.array(scores, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    error = ((scores - targets) ** 2 * sensitive).sum()
    samples = sensitive.sum()
    value = error / samples if samples else 0
    return c.Value(
        c.TargetedNumber(value, 0),
        depends=[
            quantities.samples(samples),
        ],
    )


@c.measure(
    "r2 coefficient of determination (values much smaller than zero indicate terrible models)",
    unit=False,
)
def r2(scores, targets, sensitive=None, deg_freedom=0):
    scores = np.array(scores, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    num_sensitive = sensitive.sum()

    if num_sensitive == 0:
        value = 0
    else:
        target_mean = (targets * sensitive).sum() / num_sensitive
        residual_sum = ((scores - targets) ** 2 * sensitive).sum()
        total_sum = ((targets - target_mean) ** 2 * sensitive).sum()
        if total_sum == 0:
            value = 0
        else:
            r2_value = 1 - (residual_sum / total_sum)
            if num_sensitive - 1 - deg_freedom > 0:
                r2_value *= (num_sensitive - 1) / (num_sensitive - 1 - deg_freedom)
            value = r2_value

    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.samples(num_sensitive),
            quantities.freedom(deg_freedom),
        ],
    )


@c.measure("pinball deviation", unit=False)
def pinball(scores, targets, sensitive=None, slope: float = 0.5):
    scores = np.array(scores, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)

    num_sensitive = sensitive.sum()
    loss = slope * np.maximum(targets - scores, 0) + (1 - slope) * np.maximum(
        scores - targets, 0
    )
    filtered = (loss * sensitive).sum()
    value = filtered / num_sensitive if num_sensitive else 0

    return c.Value(
        c.TargetedNumber(value, 0),
        depends=[
            quantities.samples(num_sensitive),
            quantities.slope(slope),
        ],
    )
