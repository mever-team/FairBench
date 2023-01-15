from fairbench.forks.fork import parallel_primitive
from fairbench.forks.explanation import Explainable
import inspect
from typing import Union, Iterable


def reportargsparse(*args, **kwargs):
    for arg in args:
        if not isinstance(arg, dict):
            raise TypeError(
                "Reports only support dicts of arguments as positional arguments"
            )
        for k, v in arg.items():
            if k in kwargs:
                raise TypeError(f"Report argument {k} provided multiple times")
            kwargs[k] = v
    return kwargs


@parallel_primitive
def report(*args, metrics: Union[Iterable, dict] = None, **kwargs):
    kwargs = reportargsparse(*args, **kwargs)
    if metrics is None:
        raise Exception(
            "Cannot use fairbench.report() without explicitly declared metrics.\nUse accreport, binreport, multireport, or isecreport as ad-hoc report generation mechanisms."
        )
    if not isinstance(metrics, dict):
        metrics = {metric.__name__: metric for metric in metrics}
    ret = dict()
    for name, metric in metrics.items():
        arg_names = set(inspect.getfullargspec(metric)[0])
        ret[name] = metric(
            **{
                arg: Explainable(value, desc=arg)
                for arg, value in kwargs.items()
                if arg in arg_names
            }
        )
    return ret
