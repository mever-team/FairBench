from fairbench.v2.core import report
from fairbench.v2 import blocks as blocks
from fairbench.v2.core import Sensitive, Descriptor, Progress
from fairbench.v1 import core as deprecated
import numpy as np

all_measures = [
    blocks.measures.acc,
    blocks.measures.pr,
    blocks.measures.tpr,
    blocks.measures.tnr,
    blocks.measures.ppv,
    blocks.measures.f1,
    blocks.measures.gmi,
    blocks.measures.tar,
    blocks.measures.trr,
    blocks.measures.lift,
    blocks.measures.mcc,
    blocks.measures.kappa,
    blocks.measures.avgscore,
    blocks.measures.auc,
    blocks.measures.ndcg,
    blocks.measures.topndcg,
    blocks.measures.tophr,
    blocks.measures.toprec,
    blocks.measures.topf1,
    blocks.measures.nmrr,
    blocks.measures.nentropy,
    blocks.measures.mabs,
    blocks.measures.rmse,
    blocks.measures.r2,
    blocks.measures.pinball,
    blocks.measures.spearman,
    blocks.measures.rbo,
    blocks.measures.ndrl,
]

reductions_pairwise = [
    blocks.reduction.min,
    blocks.reduction.max,
    blocks.reduction.maxerror,
    blocks.reduction.wmean,
    blocks.reduction.mean,
    blocks.reduction.gm,
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

conflate_descriptor = Descriptor(
    "conflate",
    "analysis",
    "analysis for pair of sensitive attributes",
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


def vsall(
    sensitive: Sensitive | deprecated.Fork, measures=None, reductions=None, **kwargs
):
    if measures is None:
        measures = all_measures
    if reductions is None:
        reductions = reductions_vs_any
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
        sensitive=sensitive, measures=measures, reductions=reductions, **kwargs
    )

def conflate(
    sensitive: Sensitive | deprecated.Fork, measures=None, reductions=None, **kwargs):

    if measures is None:
        measures = all_measures
    if reductions is None:
        reductions = reductions_vs_any
    # prepare the sensitive attribute, because we are going to add one more branch here
    if isinstance(sensitive, dict):
        sensitive = deprecated.Fork(sensitive)
    if isinstance(sensitive, deprecated.Fork):
        sensitive = Sensitive({k: v.numpy() for k, v in sensitive.branches().items()})
    branches = sensitive.branches
    from fairbench import Progress
    progress = Progress(conflate_descriptor.details)
    for branch1 in branches:
        branch1_progress = Progress(conflate_descriptor.details)
        for branch2 in branches:
            if branch1==branch2:
                continue
            sensitive = Sensitive({branch1: branches[branch1], branch2: branches[branch2]}, conflate_descriptor)
            conflate_report = report(
                sensitive=sensitive, measures=measures, reductions=reductions, **kwargs
            )
            branch1_progress.instance(branch2, conflate_report)
        progress.instance(branch1, branch1_progress.build())
    return progress.build()