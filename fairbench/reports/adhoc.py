from fairbench.reports.base import report, reportargsparse
from fairbench.reports import reduction as fb
from fairbench.reports.accumulate import kwargs as tokwargs
from fairbench.forks.fork import combine
from fairbench.reports.surrogate import surrogate_positives
from fairbench import metrics


common_metrics = (metrics.accuracy, metrics.prule, metrics.dfpr, metrics.dfnr)
acc_metrics = (metrics.accuracy, metrics.pr, metrics.tpr, metrics.tnr)


def accreport(*args, metrics=acc_metrics, **kwargs):
    return report(*args, metrics=metrics, **kwargs)


def binreport(*args, metrics=common_metrics, **kwargs):
    return report(*args, metrics=metrics, **kwargs)


def multireport(*args, metrics=acc_metrics, **kwargs):
    base = report(*args, metrics=metrics, **kwargs)
    return combine(
        fb.reduce(base, fb.mean),
        fb.reduce(base, fb.min, expand=fb.ratio),
        fb.reduce(base, fb.max, expand=fb.diff),
    )


def isecreport(*args, **kwargs):
    if len(args) == 0:
        params = tokwargs(**kwargs)
    else:
        params = dict()
        for arg in args:
            params = params | arg
        params = params | kwargs

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
