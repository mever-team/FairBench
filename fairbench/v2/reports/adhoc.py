from fairbench.v2.core import report
from fairbench.v2 import blocks as blocks
from fairbench.v2.core import Sensitive, Descriptor
from fairbench.v1 import core as deprecated
import numpy as np


all_measures = [
    blocks.measures.acc,
    blocks.measures.pr,
    blocks.measures.tpr,
    blocks.measures.tnr,
    blocks.measures.tar,
    blocks.measures.trr,
    blocks.measures.avgscore,
    blocks.measures.auc,
    blocks.measures.tophr,
    blocks.measures.toprec,
    blocks.measures.topf1,
    blocks.measures.mabs,
    blocks.measures.rmse,
]

reductions_pairwise = [
    blocks.reduction.min,
    blocks.reduction.max,
    blocks.reduction.maxerror,
    blocks.reduction.wmean,
    blocks.reduction.mean,
    blocks.reduction.maxbarea,
    blocks.reduction.maxrel,
    blocks.reduction.maxdiff,
    blocks.reduction.gini,
    blocks.reduction.stdx2,
]

# the following reductions should be applied only when the total population is also known
reductions_vs_any = [
    blocks.reduction.min,
    blocks.reduction.max,
    blocks.reduction.largestmaxrel,
    blocks.reduction.largestmaxdiff,
    blocks.reduction.largestmaxbarea,
]

vsall_descriptor = Descriptor(
    "vsall",
    "analysis",
    "analysis that includes the whole population ('all') to compare against",
)


def pairwise(
    sensitive: Sensitive | deprecated.Fork, measures=None, reductions=None, **kwargs
):
    if measures is None:
        measures = all_measures
    if reductions is None:
        reductions = reductions_pairwise
    return report(
        sensitive=sensitive, measures=measures, reductions=reductions, **kwargs
    )


def vsall(sensitive: Sensitive | deprecated.Fork, measures=None, **kwargs):
    if measures is None:
        measures = all_measures
    # prepare the sensitive attribute, because we are going to add one more branch here
    if isinstance(sensitive, dict):
        sensitive = deprecated.Fork(sensitive)
    if isinstance(sensitive, deprecated.Fork):
        sensitive = Sensitive({k: v.numpy() for k, v in sensitive.branches().items()})
    branches = sensitive.branches | {
        "all": np.ones_like(next(sensitive.branches.values().__iter__()))
    }
    sensitive = Sensitive(branches, vsall_descriptor)
    return report(
        sensitive=sensitive, measures=measures, reductions=reductions_vs_any, **kwargs
    )
