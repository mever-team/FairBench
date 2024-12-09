import eagerpy as ep
from typing import Iterable, Optional
from fairbench.v1.core import Explainable, ExplainableError, ExplanationCurve
import numpy as np
from functools import wraps


def expander(method):
    @wraps(method)
    def expand(
        values: Iterable[ep.Tensor], base: Iterable[ep.Tensor] = None, *args, **kwargs
    ) -> Iterable[ep.Tensor]:
        assert isinstance(values, list), "Fairbench can only expand lists."
        if base is None:
            base = values
        return method(values, base, *args, **kwargs)

    return expand


@expander
def ratio(
    values: Iterable[ep.Tensor], base: Iterable[ep.Tensor]
) -> Iterable[ep.Tensor]:
    return [value1 / value2 for value1 in values for value2 in base if value2 != 0]


@expander
def rdiff(
    values: Iterable[ep.Tensor], base: Iterable[ep.Tensor]
) -> Iterable[ep.Tensor]:
    return [
        abs(1 - value1 / value2)
        for value1 in values
        for value2 in base
        if value2 != 0 and value1 <= value2
    ]


@expander
def diff(values: Iterable[ep.Tensor], base: Iterable[ep.Tensor]) -> Iterable[ep.Tensor]:
    return [abs(value1 - value2) for value1 in values for value2 in base]


@expander
def barea(
    values: Iterable[ep.Tensor],
    base: Optional[ep.Tensor],
    skew=lambda x, y: y,
    comparator=lambda y1, y2: np.absolute(y1 - y2),
) -> Iterable[ep.Tensor]:
    assert isinstance(values, list), "fairbench.barea can only reduce lists."
    x_min = None
    x_max = None
    n_max = float("inf")
    for value in values + base:
        if (
            not isinstance(value, Explainable)
            or "curve" not in value.explain.branches()
            or not isinstance(value.explain.distribution, ExplanationCurve)
        ):
            raise ExplainableError("Explanation absent or does not store `curve`")
        if x_min is None:
            x_min = value.explain.distribution.x.min()
        elif value.explain.distribution.x.min() != x_min:
            raise ExplainableError(
                f"Incompatible curves min: {x_min} vs {value.explain.distribution.x.min()}"
            )
        if x_max is None:
            x_max = value.explain.distribution.x.max()
        elif value.explain.distribution.x.max() != x_max:
            raise ExplainableError(
                f"Incompatible curve max: {x_max} vs {value.explain.distribution.x.max()}"
            )
        n_max = min(
            n_max, value.explain.distribution.points
        )  # get the same discretization as the densest curve
    x = values[0].explain.distribution.togrid(n_max).x
    values = [skew(x, value.explain.distribution.togrid(n_max).y) for value in values]
    base = [skew(x, value.explain.distribution.togrid(n_max).y) for value in base]
    x_integral = np.mean(skew(x, np.ones_like(x)))

    return [
        np.mean(comparator(value1, value2))
        / x_integral  # TODO: find why this prompts syntax warning
        for value1 in values
        for value2 in base
    ]


def _relative(y1, y2):
    numerator = np.maximum(y1, y2) - np.minimum(y1, y2)
    denominator = np.maximum(y1, y2)
    with np.errstate(invalid="ignore"):
        result = np.divide(numerator, denominator)
        result[denominator == 0] = 0
    return result


def rarea(
    values: Optional[ep.Tensor],
    base: Optional[ep.Tensor],
):
    return barea(values, base, comparator=_relative)


def ndcg_skew(x, y):
    if x.min() < 1:
        x += 1 - x.min()
    return y / np.log(x + 1)


@expander
def bdcg(values: Iterable[ep.Tensor], base: Iterable[ep.Tensor]) -> Iterable[ep.Tensor]:
    return barea(values, base, ndcg_skew)


def kl(y1, y2):
    return y1 * np.log(y1 / (y2 + 1.0e-12) + 1.0e-12)


def js(y1, y2):
    m = (y1 + y2) / 2
    return (kl(y1, m) + kl(y2, m)) / 2


@expander
def kldcg(
    values: Iterable[ep.Tensor], base: Optional[ep.Tensor]
) -> Iterable[ep.Tensor]:
    return barea(values, base, lambda x, y: y / np.log(x + 1), comparator=kl)


@expander
def jsdcg(
    values: Iterable[ep.Tensor], base: Optional[ep.Tensor]
) -> Iterable[ep.Tensor]:
    return barea(values, base, lambda x, y: y / np.log(x + 1 + 1.0e-12), comparator=js)


@expander
def todata(
    values: Iterable[ep.Tensor], base: Optional[ep.Tensor]
) -> Iterable[ep.Tensor]:
    assert (
        base is values
    ), "Logical error: Cannot convert to data anything with a base comparative evaluation"
    values = [ep.reshape(value, (-1, 1)) for value in values]
    return ep.concatenate(values, axis=1)
