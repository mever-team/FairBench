from fairbench.fork import parallel
import inspect
from typing import Union, Iterable


@parallel
def report(*args, metrics: Union[Iterable, dict] = None, **kwargs):
    for arg in args:
        if not isinstance(arg, dict):
            raise TypeError(
                "Report can only support dicts of arguments as positional arguments"
            )
        for k, v in arg.items():
            if k in kwargs:
                raise TypeError(f"Report argument {k} provided multiple times")
            kwargs[k] = v
    if metrics is None:
        raise Exception(
            "Cannot use fairbench.report() without explicitly declared metrics.\nUse fairbench.binreport or fairbench.multireport as ad-hoc report generation mechanisms."
        )
    if not isinstance(metrics, dict):
        metrics = {metric.__name__: metric for metric in metrics}
    ret = dict()
    for name, metric in metrics.items():
        if name == "framework" or name == "parallel":
            continue
        arg_names = set(inspect.getfullargspec(metric)[0])
        try:
            ret[name] = metric(
                **{arg: value for arg, value in kwargs.items() if arg in arg_names}
            )
        except TypeError:
            pass
    return ret
