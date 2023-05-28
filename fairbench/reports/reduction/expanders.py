import eagerpy as ep
from typing import Iterable


def ratio(values: Iterable[ep.Tensor]) -> Iterable[ep.Tensor]:
    assert isinstance(values, list), "Can only reduce lists with fairbench.ratio."
    return [value1 / value2 for value1 in values for value2 in values if value2 != 0]


def diff(values: Iterable[ep.Tensor]) -> Iterable[ep.Tensor]:
    assert isinstance(values, list), "Can only reduce lists with fairbench.diff."
    return [abs(value1 - value2) for value1 in values for value2 in values]


def todata(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(
        values, list
    ), "Can only reduce lists of tensors with fairbench.todata."
    values = [ep.reshape(value, (-1, 1)) for value in values]
    return ep.concatenate(values, axis=1)
