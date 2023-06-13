import eagerpy as ep
from typing import Iterable


def abs(value):
    if value < 0:
        return -value
    return value


def sum(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "Can only reduce lists with fairbench.sum. Maybe you meant to use eagerpy.sum?"
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
    assert isinstance(
        values, list
    ), "Can only reduce lists with fairbench.mean. Maybe you meant to use eagerpy.identical?"
    for value in values:
        assert (
            value - values[0]
        ).abs().sum() == 0, "eagerpy.identical requires that the exact same tensor is placed on all branches"
    return values[0]


def gm(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "Can only reduce lists with fairbench.mean. Maybe you meant to use eagerpy.mean?"
    ret = 1
    for value in values:
        ret = ret * value
    return ret ** (1.0 / len(values))


def max(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "Can only reduce lists with fairbench.max. Maybe you meant to use eagerpy.maximum?"
    ret = float("-inf")
    for value in values:
        if value > ret:
            ret = value
    return ret


def budget(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "Can only reduce lists with fairbench.budget. Maybe you meant to use an eagerpy method?"
    from math import log  # TODO: make this compatible with backpropagation

    # "An Intersectional Definition of Fairness"
    return max(values).log()


def min(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "Can only reduce lists with fairbench.min. Maybe you meant to use eagerpy.minimum?"
    ret = float("inf")
    for value in values:
        if value < ret:
            ret = value
    return ret
