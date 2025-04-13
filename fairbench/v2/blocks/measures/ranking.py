from fairbench.v2 import core as c
from fairbench.v2.blocks.quantities import quantities
import numpy as np


@c.measure("the Spearman correlation", unit=False)
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
def rbo(scores, order, sensitive=None, top_weightedness=1.0):
    from fairbench.fallbacks.rbo import RankingSimilarity

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
    value = RankingSimilarity(scores, order).rbo(p=top_weightedness)
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[
            quantities.samples(samples),
        ],
    )


@c.measure("the normalized discounted ranking loss", unit=False)
def ndrl(scores, order, sensitive=None):
    scores = np.array(scores, dtype=np.float64)
    order = np.array(order, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)

    scores = scores[sensitive > 0]
    order = order[sensitive > 0]
    samples = sensitive.sum()

    if len(scores) == 0 or len(order) == 0:
        return c.Value(c.TargetedNumber(0, 0), depends=[])

    pred_ranks = np.argsort(np.argsort(-scores))
    true_ranks = np.argsort(np.argsort(-order))

    n = len(scores)
    discount = np.log2(np.arange(2, n + 2))

    # Per-rank discounted loss
    per_item_loss = np.abs(pred_ranks - true_ranks) / discount
    discounted_loss = per_item_loss.sum()

    # WORST prediction: sort true ranks ascending
    worst_pred = np.argsort(true_ranks)
    max_per_item_loss = np.abs(worst_pred - true_ranks) / discount
    max_loss = max_per_item_loss.sum()

    value = 0.0 if max_loss == 0 else discounted_loss / max_loss
    loss_curve = c.Curve(
        x=np.arange(1, n + 1, dtype=float) / n,
        y=per_item_loss,
        units="",
    )

    return c.Value(
        c.TargetedNumber(value, 0),
        depends=[
            quantities.samples(samples),
            quantities.itemloss(loss_curve),
        ],
    )


@c.measure("the normalized discounted cumulative gain of all recommendations")
def ndcg(scores, order, sensitive=None):
    scores = np.array(scores, dtype=np.float64)
    order = np.array(order, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    scores = scores[sensitive > 0]
    order = order[sensitive > 0]
    samples = sensitive.sum()
    assert samples != 0, f"Cannot compute NDCG for an empty group"

    indexes = np.argsort(scores)[::-1]
    rel = order[indexes]
    dcg = np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
    ideal_rel = np.sort(order)[::-1]
    idcg = np.sum((2**ideal_rel - 1) / np.log2(np.arange(2, len(ideal_rel) + 2)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    true_top = order.sum()
    return c.Value(
        ndcg,
        depends=[
            quantities.tp(true_top),
            quantities.samples(samples),
        ],
    )


@c.measure("the normalized discounted cumulative gain of top recommendations")
def topndcg(scores, order, sensitive=None, top=3):
    scores = np.array(scores, dtype=np.float64)
    order = np.array(order, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    samples = sensitive.sum()

    k = int(top)
    assert (
        0 < k <= scores.shape[0]
    ), f"There are only {scores.shape[0]} inputs but top={top} was requested for ranking analysis"
    assert samples != 0, f"Cannot compute topndfcg for an empty group"

    scores = scores[sensitive > 0]
    order = order[sensitive > 0]

    indexes = np.argsort(scores)[-k:][::-1]
    rel = order[indexes]
    dcg = np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
    ideal_rel = np.sort(order)[-k:][::-1]
    idcg = np.sum((2**ideal_rel - 1) / np.log2(np.arange(2, len(ideal_rel) + 2)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    true_top = order[indexes].sum()

    return c.Value(
        ndcg,
        depends=[
            quantities.top(k),
            quantities.tp(true_top),
            quantities.samples(samples),
        ],
    )
