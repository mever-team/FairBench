from fairbench.v1.core import parallel_primitive, comparator, role, Fork
from fairbench.v1.core import Explainable
import inspect
from typing import Union, Iterable, Callable


def reportargsparse(*args, **kwargs):
    for arg in args:
        assert isinstance(arg, dict) or isinstance(
            arg, Fork
        ), "Reports only support dicts of arguments as positional arguments"
        for k, v in arg.items():
            assert k not in kwargs, f"Report argument {k} provided multiple times"
            kwargs[k] = v
    return kwargs


def report(
    *args, metrics: Union[Callable, Iterable, dict] = None, modify_kwargs=None, **kwargs
):
    """
    This is a wrapper method of the true _report method, which also adds the prospect of modifying kwargs.
    """
    kwargs = reportargsparse(*args, **kwargs)
    if modify_kwargs is not None:
        kwargs = modify_kwargs(kwargs)
    return _report(metrics, **kwargs)


@role("report")
@comparator
@parallel_primitive
def _report(metrics: Union[Callable, Iterable, dict] = None, **kwargs):
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
