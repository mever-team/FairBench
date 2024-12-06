from fairbench.experimental.core_v2 import report
from fairbench.experimental import blocks_v2 as blocks
from fairbench.experimental.core_v2 import Sensitive
from fairbench import core as deprecated

measures = [
    blocks.measures.acc,
    blocks.measures.pr,
    blocks.measures.tpr,
    blocks.measures.tnr,
    blocks.measures.tar,
    blocks.measures.trr,
    blocks.measures.mabs,
    blocks.measures.rmse,
    blocks.measures.pinball,
]

reductions = [
    blocks.reduction.min,
    blocks.reduction.wmean,
    blocks.reduction.maxrel,
    blocks.reduction.maxdiff,
    blocks.reduction.std,
]

def pairwise(sensitive: Sensitive | deprecated.Fork, **kwargs):
    return report(sensitive=sensitive, measures=measures, reductions=reductions, **kwargs)
