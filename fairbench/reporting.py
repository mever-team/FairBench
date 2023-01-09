from fairbench.modal import multimodal
from inspect import getmembers, isfunction
import inspect
from fairbench import metrics as metric_package
from types import MappingProxyType

_found_metrics = MappingProxyType(
    dict(getmembers(metric_package, isfunction))
)  # creates an immutable map


@multimodal
def report(metrics=_found_metrics, **kwargs):
    ret = dict()
    for name, metric in metrics.items():
        if name == "framework" or name == "multimodal":
            continue
        arg_names = set(inspect.getfullargspec(metric)[0])
        try:
            ret[name] = metric(
                **{arg: value for arg, value in kwargs.items() if arg in arg_names}
            )
        except TypeError:
            pass
    return ret
