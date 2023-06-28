from fairbench.reports.base import report
from fairbench.reports import reduction as fb
from fairbench.reports.accumulate import todict as tokwargs
from fairbench.forks.fork import combine, merge, role
from fairbench.reports.surrogate import surrogate_positives
from fairbench import metrics
from fairbench.reports import reduction


common_metrics = (metrics.accuracy, metrics.prule, metrics.dfpr, metrics.dfnr)
acc_metrics = (
    metrics.accuracy,
    metrics.pr,
    metrics.tpr,
    metrics.tnr,
    metrics.auc,
    metrics.r2,
)
common_reduction = (
    {"reducer": reduction.min},
    {"reducer": reduction.wmean},
    {"reducer": reduction.min, "expand": reduction.ratio},
    {"reducer": reduction.max, "expand": reduction.diff},
    {"reducer": reduction.max, "expand": reduction.barea},
)


def accreport(*args, metrics=acc_metrics, **kwargs):
    return report(*args, metrics=metrics, **kwargs)


def binreport(*args, metrics=common_metrics, **kwargs):
    return report(*args, metrics=metrics, **kwargs)


@role("report")
def multireport(
    *args, metrics=acc_metrics, reduction_schemes=common_reduction, **kwargs
):
    base = report(*args, metrics=metrics, **kwargs)
    return combine(*[fb.reduce(base, **scheme) for scheme in reduction_schemes])


@role("report")
def isecreport(*args, **kwargs):
    if len(args) == 0:
        params = tokwargs(**kwargs)
    else:
        params = dict()
        for arg in args:
            params = merge(params, arg)
        params = merge(params, kwargs)

    bayesian = fb.reduce(
        surrogate_positives(params["predictions"], params["sensitive"]),
        fb.min,
        fb.ratio,
        name="bayesian",
    )

    empirical = fb.reduce(
        metrics.pr(predictions=params["predictions"], sensitive=params["sensitive"]),
        fb.min,
        fb.ratio,
        name="empirical",
    )
    return combine(tokwargs(minprule=empirical), tokwargs(minprule=bayesian))
