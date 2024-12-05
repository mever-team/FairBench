from fairbench.experimental import core_v2 as c
from fairbench.experimental.blocks_v2.quantities import quantities
import numpy as np


@c.measure("mean absolute error")
def mabs(scores, targets, sensitive=None):
    scores = np.array(scores, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    error = (np.abs(scores-targets)*sensitive).sum()
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
    error = ((scores-targets)**2 * sensitive).sum()
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
    error = ((scores-targets)**2 * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0.0 else error / samples
    return c.Value(
        c.TargetedNumber(value, 0),
        depends=[
            quantities.samples(samples),
        ],
    )

@c.measure("coefficient of determination", unit=False)
def r2(scores, targets, sensitive=None, deg_freedom=0):
    scores = np.array(scores, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    num_sensitive = sensitive.sum()
    true = ((scores - targets) ** 2 * sensitive).sum()
    target_mean_squares = (targets**2 * sensitive).sum() / num_sensitive
    target_mean = (targets**2 * sensitive).sum() / num_sensitive
    target_variance = target_mean_squares - target_mean**2
    value = (
        0
        if num_sensitive == 0
        else (1 - (true / target_variance)) * ((num_sensitive - 1) / (num_sensitive - 1 - deg_freedom))
    )
    return c.Value(
        c.TargetedNumber(value, 0),
        depends=[
            quantities.samples(num_sensitive),
            quantities.freedom(deg_freedom),
        ],
    )

@c.measure("pinball deviation")
def pinball(scores, targets, sensitive=None, slope: float = 0.5):
    scores = np.array(scores, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    num_sensitive = sensitive.sum()
    loss = slope * np.max(targets - scores) + (1 - slope) * np.max(scores - targets)
    filtered = (loss * sensitive).sum()
    value = 0 if num_sensitive == 0 else filtered / num_sensitive
    return c.Value(
        c.TargetedNumber(value, 0),
        depends=[
            quantities.samples(num_sensitive),
            quantities.slope(slope),
        ],
    )