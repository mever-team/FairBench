from fairbench.v2 import core as c
from fairbench.v2.blocks.quantities import quantities
import numpy as np


@c.measure("the average score")
def avgscore(scores, sensitive=None, bins=100):
    scores = np.array(scores)
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
        np.array((bin_edges[:-1] + bin_edges[1:]) / 2, dtype=float),
        np.array(hist, dtype=float),
        "Prob. density",
    )

    return c.Value(
        value,
        depends=[
            quantities.positives(positives),
            quantities.samples(samples),
            quantities.distribution(curve),
        ],
    )
