from fairbench import metrics
from fairbench.utils import missing
from inspect import getmembers, isfunction


def report(yhat, **kwargs):
    ret = ""
    for name, metric in getmembers(metrics, isfunction):
        if name == "framework":
            continue
        eval = metric(yhat)
        miss = missing(eval, **kwargs)
        if miss:
            ret += f"{name.ljust(10)}\t NA (no {','.join(miss)})\n"
        else:
            ret += f"{name.ljust(10)}\t {eval(**kwargs):.3f}\n"
    return ret
