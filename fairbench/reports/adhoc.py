from fairbench.reports.base import report
from fairbench.reports import reduction as fb
from fairbench.reports.accumulate import kwargs as tokwargs
from fairbench.forks.fork import combine
from fairbench.reports.surrogate import surrogate_positives
from fairbench import metrics


def accreport(*args, **kwargs):
    return report(
        *args,
        metrics=(metrics.accuracy, metrics.pr, metrics.tpr, metrics.tnr),
        **kwargs
    )


def binreport(*args, **kwargs):
    return report(
        *args,
        metrics=(metrics.accuracy, metrics.prule, metrics.dfpr, metrics.dfnr),
        **kwargs
    )


def multireport(*args, **kwargs):
    base = report(
        *args,
        metrics=(metrics.accuracy, metrics.pr, metrics.tpr, metrics.tnr),
        **kwargs
    )
    return combine(
        fb.reduce(base, fb.mean),
        fb.reduce(base, fb.min, expand=fb.ratio),
        fb.reduce(base, fb.max, expand=fb.diff),
    )


def isecreport(*args, **kwargs):
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
