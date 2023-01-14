from fairbench.fork import parallel
from inspect import getmembers, isfunction
import inspect
from fairbench import metrics as metric_package
from types import MappingProxyType

_found_metrics = MappingProxyType(
    dict(getmembers(metric_package, isfunction))
)  # creates an immutable map


@parallel
def report(*args, metrics=_found_metrics, **kwargs):
    for arg in args:
        if not isinstance(arg, dict):
            raise TypeError("Report can only support dicts of arguments as positional arguments")
        for k, v in arg.items():
            if k in kwargs:
                raise TypeError(f"Report argument {k} provided multiple times")
            kwargs[k] = v

    print(kwargs)

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
