from fairbench import metrics
from fairbench.utils import missing, framework
from inspect import getmembers, isfunction
import pyfop as pfp


@framework
def report(yhat, **kwargs):
    ret = ""
    for name, metric in getmembers(metrics, isfunction):
        if name == "framework":
            continue
        eval = metric(yhat)
        complete = {key: kwargs[key] for key in missing(eval) if key in kwargs}
        miss = missing(eval, **complete)
        if miss:
            ret += f"{name.ljust(10)}\t NA (no {','.join(miss)})\n"
        else:
            ret += f"{name.ljust(10)}\t {eval(**complete):.3f}\n"
    return ret
