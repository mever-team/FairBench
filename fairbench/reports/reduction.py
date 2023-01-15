from fairbench.fork import Fork
from typing import Iterable
import eagerpy as ep


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


def reduce(fork: Fork, method, expand=None, transform=None, branches=None, name=None):
    if name is None:
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
                fields[f].append((v[f]) if transform is None else transform((v[f])))
        else:
            fields.append((v) if transform is None else transform((v[v])))
    if expand is not None:
        fields = (
            {k: expand(v) for k, v in fields.items()}
            if isinstance(fields, dict)
            else expand(fields)
        )
    result = (
        {k: (method(v)) for k, v in fields.items()}
        if isinstance(fields, dict)
        else method(fields)
    )
    return Fork({name: result})
