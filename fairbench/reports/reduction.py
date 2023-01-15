from fairbench.forks.fork import Fork, astensor
from typing import Iterable
import eagerpy as ep
from fairbench.forks.explanation import Explainable

# from fairbench.reports.accumulate import kwargs as tokwargs


def abs(value):
    if value < 0:
        return -value
    return value


def sum(values: Iterable[ep.Tensor]) -> ep.Tensor:
    if not isinstance(values, list):
        raise TypeError(
            "Can only reduce lists with fairbench.sum. Maybe you meant to use eagerpy.sum?"
        )
    ret = 0
    for value in values:
        ret = ret + value
    return ret


def mean(values: Iterable[ep.Tensor]) -> ep.Tensor:
    if not isinstance(values, list):
        raise TypeError(
            "Can only reduce lists with fairbench.mean. Maybe you meant to use eagerpy.mean?"
        )
    return sum(values) / len(values)


def identical(values: Iterable[ep.Tensor]) -> ep.Tensor:
    if not isinstance(values, list):
        raise TypeError(
            "Can only reduce lists with fairbench.mean. Maybe you meant to use eagerpy.identical?"
        )
    for value in values:
        if (value - values[0]).abs().sum() != 0:
            raise Exception(
                "eagerpy.identical requires that the exact same tensor is placed on all branches"
            )
    return values[0]


def gm(values: Iterable[ep.Tensor]) -> ep.Tensor:
    if not isinstance(values, list):
        raise TypeError(
            "Can only reduce lists with fairbench.mean. Maybe you meant to use eagerpy.mean?"
        )

    ret = 1
    for value in values:
        ret = ret * value
    return ret ** (1.0 / len(values))


def max(values: Iterable[ep.Tensor]) -> ep.Tensor:
    if not isinstance(values, list):
        raise TypeError(
            "Can only reduce lists with fairbench.max. Maybe you meant to use eagerpy.maximum?"
        )
    ret = float("-inf")
    for value in values:
        if value > ret:
            ret = value
    return ret


def budget(values: Iterable[ep.Tensor]) -> ep.Tensor:
    if not isinstance(values, list):
        raise TypeError(
            "Can only reduce lists with fairbench.min. Maybe you meant to use eagerpy.min?"
        )
    from math import log  # TODO: make this compatible with backpropagation

    # "An Intersectional Definition of Fairness"
    return log(float(max(values)))


def min(values: Iterable[ep.Tensor]) -> ep.Tensor:
    if not isinstance(values, list):
        raise TypeError(
            "Can only reduce lists with fairbench.min. Maybe you meant to use eagerpy.min?"
        )
    ret = float("inf")
    for value in values:
        if value < ret:
            ret = value
    return ret


def ratio(values: Iterable[ep.Tensor]) -> Iterable[ep.Tensor]:
    if not isinstance(values, list):
        raise TypeError("Can only reduce lists with fairbench.ratio.")
    return [value1 / value2 for value1 in values for value2 in values if value2 != 0]


def diff(values: Iterable[ep.Tensor]) -> Iterable[ep.Tensor]:
    if not isinstance(values, list):
        raise TypeError("Can only reduce lists with fairbench.diff.")
    return [abs(value1 - value2) for value1 in values for value2 in values]


def todata(values: Iterable[ep.Tensor]) -> ep.Tensor:
    if not isinstance(values, list):
        raise TypeError("Can only reduce lists of tensors with fairbench.todata.")
    values = [ep.reshape(value, (-1, 1)) for value in values]
    return ep.concatenate(values, axis=1)


def reduce(fork: Fork, method, expand=None, transform=None, branches=None, name=""):
    if name == "":
        name = method.__name__
        if expand is not None:
            name += expand.__name__
        if transform is not None:
            name += transform.__name__
        if branches is not None:
            name += "[" + ",".join(branches) + "]"
    fields = None
    for branch, v in fork.branches().items():
        if branches is not None and branch not in branches:
            continue
        if fields is None:
            fields = {f: list() for f in v} if isinstance(v, dict) else list()
        if isinstance(v, dict):
            for f in v:
                fields[f].append(
                    astensor(v[f]) if transform is None else transform(astensor(v[f]))
                )
        else:
            fields.append(
                astensor(v) if transform is None else transform(astensor(v[v]))
            )
    if expand is not None:
        fields = (
            {k: expand(v) for k, v in fields.items()}
            if isinstance(fields, dict)
            else expand(fields)
        )
    result = (
        {k: Explainable(method(v), fork, desc=name) for k, v in fields.items()}
        if isinstance(fields, dict)
        else Explainable(method(fields), fork, desc=name)
    )
    return result if name is None else Fork({name: result})
