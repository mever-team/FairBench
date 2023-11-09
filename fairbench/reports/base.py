from fairbench.core.fork import parallel_primitive, comparator, role
from fairbench.core.explanation import Explainable
import inspect
from typing import Union, Iterable, Callable


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


@role("report")
@comparator
@parallel_primitive
def report(*args, metrics: Union[Callable, Iterable, dict] = None, **kwargs):
    kwargs = reportargsparse(*args, **kwargs)
    assert (
        metrics is not None
    ), "Cannot use fairbench.report() without explicitly declared metrics.\nUse accreport, binreport, multireport, or isecreport as ad-hoc report generation mechanisms."
    if not isinstance(metrics, Iterable):
        metrics = [metrics]
    if not isinstance(metrics, dict):
        metrics = {metric.__name__: metric for metric in metrics}
    ret = dict()
    for name, metric in metrics.items():
        arg_names = set(inspect.getfullargspec(metric)[0])
        parsed_kwargs = {
            arg: value  # TODO: find a way to add this Explainable(value, desc=arg) - this makes measures compute on explainable objects, which throws an error
            for arg, value in kwargs.items()
            if arg in arg_names
            and (
                not isinstance(value, str) or arg != value
            )  # last statement for yamlres support
        }
        try:
            ret[name] = metric(**parsed_kwargs)
        except TypeError:
            pass
    return ret


def areport(*args, metrics: Union[Callable, Iterable, dict] = None, **kwargs):
    assert (
        metrics is not None
    ), "Cannot use fairbench.report() without explicitly declared metrics.\nUse accreport, binreport, multireport, or isecreport as ad-hoc report generation mechanisms."
    if not isinstance(metrics, Iterable):
        return getattr(report(*args, metrics=[metrics], **kwargs), metrics.__name__)
    if not isinstance(metrics, dict):
        return [
            getattr(report(*args, metrics=[metric], **kwargs), metric.__name__)
            for metric in metrics
        ]
    return {
        name: getattr(report(*args, metrics={name: metric}, **kwargs), name)
        for name, metric in metrics.items()
    }
