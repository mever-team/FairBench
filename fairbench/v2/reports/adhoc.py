from fairbench.v2.core import report
from fairbench.v2 import blocks as blocks
from fairbench.v2.core import Sensitive, Descriptor
from fairbench.v1 import core as deprecated
import numpy as np


measures = [
    blocks.measures.acc,
    blocks.measures.pr,
    blocks.measures.tpr,
    blocks.measures.tnr,
    blocks.measures.tar,
    blocks.measures.trr,
    # blocks.measures.avgscore,
    blocks.measures.mabs,
    blocks.measures.rmse,
]

reductions = [
    blocks.reduction.min,
    blocks.reduction.max,
    blocks.reduction.wmean,
    blocks.reduction.maxrel,
    blocks.reduction.maxdiff,
    blocks.reduction.std,
]

# the following reductions should be applied only when the total population is also known
reductions_vs_any = [
    blocks.reduction.min,
    blocks.reduction.max,
    blocks.reduction.wmean,
    blocks.reduction.largestmaxrel,
    blocks.reduction.largestmaxdiff,
]

vsall_descriptor = Descriptor(
    "vsall",
    "analysis",
    "analysis that includes the whole population ('all') to compare against",
)


def pairwise(sensitive: Sensitive | deprecated.Fork, **kwargs):
    return report(
        sensitive=sensitive, measures=measures, reductions=reductions, **kwargs
    )


def vsall(sensitive: Sensitive | deprecated.Fork, **kwargs):
    # prepare the sensitive attribute, because we are going to add one more branch here
    if isinstance(sensitive, deprecated.Fork):
        sensitive = Sensitive({k: v.numpy() for k, v in sensitive.branches().items()})
    branches = sensitive.branches | {
        "all": np.ones_like(next(sensitive.branches.values().__iter__()))
    }
    sensitive = Sensitive(branches, vsall_descriptor)
    return report(
        sensitive=sensitive, measures=measures, reductions=reductions_vs_any, **kwargs
    )
