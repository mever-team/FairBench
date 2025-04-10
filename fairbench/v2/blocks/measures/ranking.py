from fairbench.v2 import core as c
from fairbench.v2.blocks.quantities import quantities
import numpy as np


@c.measure("the Spearman correlation")
def spearman(scores, order, sensitive=None):
    scores = np.array(scores, dtype=np.float64)
    order = np.array(order, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    scores = scores[sensitive > 0]
    order = order[sensitive > 0]
    samples = sensitive.sum()

    if scores.size == 0 or order.size == 0:
        return c.Value(
            c.TargetedNumber(0, 1),
            depends=[
                quantities.samples(samples),
            ],
        )

    pred_ranks = np.argsort(np.argsort(-scores))
    true_ranks = np.argsort(np.argsort(-order))

    d = pred_ranks - true_ranks
    d_squared = (d**2).sum()
    n = len(scores)

    denominator = n * (n**2 - 1)
    value = 1 - 6 * d_squared / denominator if denominator != 0 else 0.0

    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.samples(samples),
        ],
    )


@c.measure("the rank-biased overlap")
def rbo(scores, order, sensitive=None, p=0.9):
    scores = np.array(scores, dtype=np.float64)
    order = np.array(order, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    scores = scores[sensitive > 0]
    order = order[sensitive > 0]
    samples = sensitive.sum()

    if scores.size == 0 or order.size == 0:
        return c.Value(
            c.TargetedNumber(0, 1),
            depends=[
                quantities.samples(samples),
            ],
        )

    pred_ranking = np.argsort(scores)[::-1]
    true_ranking = np.argsort(order)[::-1]

    def _rbo(pred, true, p):
        rbo_score = 0.0
        depth = min(len(pred), len(true))
        agreement = 0
        for d in range(1, depth + 1):
            agreement += len(set(pred[:d]) & set(true[:d])) / d
            rbo_score += (p ** (d - 1)) * (agreement / d)
        return (1 - p) * rbo_score

    value = _rbo(pred_ranking, true_ranking, p)
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.samples(samples),
        ],
    )


@c.measure("the normalized discounted ranking loss")
def ndrl(scores, order, sensitive=None):
    scores = np.array(scores, dtype=np.float64)
    order = np.array(order, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    scores = scores[sensitive > 0]
    order = order[sensitive > 0]
    samples = sensitive.sum()

    if len(scores) == 0 or len(order) == 0:
        return c.Value(
            c.TargetedNumber(0, 0),
            depends=[
                quantities.samples(samples),
            ],
        )

    pred_ranks = np.argsort(np.argsort(-scores))
    true_ranks = np.argsort(np.argsort(-order))

    discounted_loss = np.sum(
        np.abs(pred_ranks - true_ranks) / np.log2(np.arange(2, len(scores) + 2))
    )

    worst_pred = true_ranks[::-1]
    max_loss = np.sum(
        np.abs(worst_pred - true_ranks) / np.log2(np.arange(2, len(scores) + 2))
    )

    value = 0.0 if max_loss == 0 else discounted_loss / max_loss
    return c.Value(
        c.TargetedNumber(value, 0),
        depends=[
            quantities.samples(samples),
        ],
    )
