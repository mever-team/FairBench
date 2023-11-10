from fairbench.reports.base import report
from fairbench.blocks import framework, reducers, expanders, metrics
from fairbench.reports.accumulate import todict as tokwargs
from fairbench.core.fork import combine, merge, role
from fairbench.reports.surrogate import surrogate_positives

common_metrics = (
    metrics.accuracy,
    metrics.prule,
    metrics.dfpr,
    metrics.dfnr,
)

acc_metrics = (
    metrics.accuracy,
    metrics.pr,
    metrics.tpr,
    metrics.tnr,
    metrics.auc,
    metrics.phi,
    metrics.hr,
    metrics.reck,
    metrics.ap,
    metrics.arepr,
    metrics.r2,
)

common_reduction = (
    {"reducer": reducers.min},
    {"reducer": reducers.wmean},
    {"reducer": reducers.min, "expand": expanders.ratio},
    {"reducer": reducers.max, "expand": expanders.diff},
    {"reducer": reducers.max, "expand": expanders.barea},
    {"reducer": reducers.max, "expand": expanders.bdcg},
    {"reducer": reducers.max, "expand": expanders.jsdcg},
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
    return combine(*[framework.reduce(base, **scheme) for scheme in reduction_schemes])


@role("report")
def isecreport(*args, **kwargs):
    if len(args) == 0:
        params = tokwargs(**kwargs)
    else:
        params = dict()
        for arg in args:
            params = merge(params, arg)
        params = merge(params, kwargs)

    bayesian = framework.reduce(
        surrogate_positives(params["predictions"], params["sensitive"]),
        reducers.min,
        expanders.ratio,
        name="bayesian",
    )

    empirical = framework.reduce(
        metrics.pr(predictions=params["predictions"], sensitive=params["sensitive"]),
        reducers.min,
        expanders.ratio,
        name="empirical",
    )
    return combine(tokwargs(minprule=empirical), tokwargs(minprule=bayesian))
