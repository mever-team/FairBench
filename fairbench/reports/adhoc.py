from fairbench.reports.base import report
from fairbench.reports import reduction as fb
from fairbench.fork import combine
from fairbench import metrics


def accreport(*args, **kwargs):
    return report(*args, metrics=(metrics.accuracy, metrics.fpr, metrics.fnr), **kwargs)


def binreport(*args, **kwargs):
    return report(
        *args,
        metrics=(metrics.accuracy, metrics.prule, metrics.dfpr, metrics.dfnr),
        **kwargs
    )


def multireport(*args, **kwargs):
    ret = report(
        *args,
        metrics=(metrics.accuracy, metrics.pr, metrics.fpr, metrics.fnr),
        **kwargs
    )
    return combine(
        fb.reduce(ret, fb.mean),
        fb.reduce(ret, fb.min, expand=fb.ratio),
        fb.reduce(ret, fb.max, expand=fb.diff),
    )
