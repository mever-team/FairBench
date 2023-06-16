import eagerpy as ep
from typing import Iterable
from fairbench.forks.explanation import Explainable, ExplainableError, ExplanationCurve
import numpy as np


def ratio(values: Iterable[ep.Tensor]) -> Iterable[ep.Tensor]:
    assert isinstance(values, list), "fairbench.ratio can only reduce lists ."
    return [value1 / value2 for value1 in values for value2 in values if value2 != 0]


def diff(values: Iterable[ep.Tensor]) -> Iterable[ep.Tensor]:
    assert isinstance(values, list), "fairbench.diff can only reduce lists."
    return [abs(value1 - value2) for value1 in values for value2 in values]


def barea(values: Iterable[ep.Tensor]) -> Iterable[ep.Tensor]:
    assert isinstance(values, list), "fairbench.diff can only reduce lists."
    x_min = None
    x_max = None
    n_max = float("inf")
    for value in values:
        if (
            not isinstance(value, Explainable)
            or "curve" not in value.explain.branches()
            or not isinstance(value.explain.curve, ExplanationCurve)
        ):
            raise ExplainableError("Explanation absent or does not store `curve`")
        if x_min is None:
            x_min = value.explain.curve.x.min()
        elif value.explain.curve.x.min() != x_min:
            raise ExplainableError(
                f"Incompatible curves min: {x_min} vs {value.explain.curve.x.min()}"
            )
        if x_max is None:
            x_max = value.explain.curve.x.max()
        elif value.explain.curve.x.max() != x_max:
            raise ExplainableError(
                f"Incompatible curve max: {x_max} vs {value.explain.curve.x.max()}"
            )
        n_max = min(
            n_max, value.explain.curve.points
        )  # get the same discretization as the densest curve
    values = [value.explain.curve.togrid(n_max).y for value in values]
    return [
        np.absolute(value1 - value2).mean() * (x_max - x_min)
        for value1 in values
        for value2 in values
    ]


def todata(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(values, list), "fairbench.todata can only reduce lists ."
    values = [ep.reshape(value, (-1, 1)) for value in values]
    return ep.concatenate(values, axis=1)
