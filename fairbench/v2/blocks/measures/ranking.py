from torch.distributed.tensor import zeros

from fairbench.v2 import core as c
from fairbench.v2.blocks.quantities import quantities
import numpy as np


@c.measure("the average score")
def avgscore(scores, sensitive=None, bins=100):
    scores = np.array(scores, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)
    positives = (scores * sensitive).sum()
    samples = sensitive.sum()
    value = 0 if samples == 0 else positives / samples

    hist, bin_edges = np.histogram(
        scores[sensitive == 1], bins=bins, density=True, range=(0, 1)
    )
    bin_edges = np.concatenate([[0], bin_edges[:-1][hist != 0], [bin_edges[-1], 1]])
    hist = np.concatenate([[0], hist[hist != 0], [0]])

    curve = c.Curve(
        x=np.array((bin_edges[:-1] + bin_edges[1:]) / 2, dtype=float),
        y=np.array(hist, dtype=float),
        units="",
    )

    return c.Value(
        value,
        depends=[
            quantities.positives(positives),
            quantities.samples(samples),
            quantities.distribution(curve),
        ],
    )


@c.measure("the area under curve of the receiver operating characteristics")
def auc(scores, labels, sensitive=None):
    from fairbench.fallbacks import auc as _auc, roc_curve as _roc_curve
    import math

    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    sensitive = np.ones_like(scores) if sensitive is None else np.array(sensitive)

    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    fpr, tpr, _ = _roc_curve(labels, scores)
    value = _auc(fpr, tpr)

    assert not math.isnan(
        value
    ), f"Cannot compute AUC when all instances have the same label for branch"

    curve = c.Curve(
        x=fpr,
        y=tpr,
        units="",
    )

    return c.Value(
        value,
        depends=[quantities.samples(sensitive.sum()), quantities.roc(curve)],
    )
