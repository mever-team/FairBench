import eagerpy as ep
from typing import Iterable
from fairbench.core import Explainable, ExplainableError


def abs(value):
    if value < 0:
        return -value
    return value


def _sum(values):
    ret = 0
    for value in values:
        ret = value + ret
    return ret


def sum(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "fairbench.sum can only reduce lists. Maybe you meant to use eagerpy.sum?"
    ret = 0
    for value in values:
        ret = ret + value
    return ret


def mean(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "fairbench.mean can only reduce lists. Maybe you meant to use eagerpy.mean?"
    return _sum(values) / len(values)


def wmean(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(values, list), "fairbench.wmean can only reduce lists."
    for value in values:
        if (
            not isinstance(value, Explainable)
            or "samples" not in value.explain.branches()
        ):
            raise ExplainableError("Explanation absent or does not store `samples`")
    nom = _sum([value * value.explain.samples for value in values])
    denom = _sum([value.explain.samples for value in values])
    return nom if denom == 0 else nom / denom


def identical(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "Can only reduce lists with fairbench.identical. Maybe you meant to use an eagerpy method?"
    for value in values:
        if (value - values[0]).abs().sum() != 0:
            raise ExplainableError(
                "The same value should reside in all branches for identical reduction."
            )
    return values[0]


def gm(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(values, list), "fairbench.gm can only reduce lists."
    ret = 1
    for value in values:
        ret = ret * value
    return ret ** (1.0 / len(values))


def max(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "fairbench.max can only reduce lists. Maybe you meant to use eagerpy.maximum?"
    ret = float("-inf")
    for value in values:
        if value > ret:
            ret = value
    return ret


def budget(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(values, list), "fairbench.budget can only reduce lists."
    from math import log  # TODO: make this compatible with backpropagation

    # "An Intersectional Definition of Fairness"
    return max(values).log()


def min(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "fairbench.min can only reduce lists. Maybe you meant to use eagerpy.minimum?"
    ret = float("inf")
    for value in values:
        if value < ret:
            ret = value
    return ret
