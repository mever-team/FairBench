import eagerpy as ep
from fairbench.v1.core import verify
from fairbench.v1.core import Explainable, ExplainableError
from typing import List


def _sum(values):
    ret = 0
    for value in values:
        ret = value + ret
    return ret


def min(values: List[ep.Tensor]) -> ep.Tensor:
    verify(
        isinstance(values, list),
        "fairbench.min can only reduce lists. Maybe you meant to use eagerpy.minimum?",
    )
    ret = float("inf")
    for value in values:
        if value < ret:
            ret = value
    return ret


def sum(values: List[ep.Tensor]) -> ep.Tensor:
    verify(
        isinstance(values, list),
        "fairbench.sum can only reduce lists. Maybe you meant to use eagerpy.sum?",
    )
    ret = 0
    for value in values:
        ret = ret + value
    return ret


def gm(values: List[ep.Tensor]) -> ep.Tensor:
    verify(isinstance(values, list), "fairbench.gm can only reduce lists.")
    ret = 1
    for value in values:
        ret = ret * value
    return ret ** (1.0 / len(values))


def mean(values: List[ep.Tensor]) -> ep.Tensor:
    verify(
        isinstance(values, list),
        "fairbench.mean can only reduce lists. Maybe you meant to use eagerpy.mean?",
    )
    return _sum(values) / len(values)


def wmean(values: List[ep.Tensor]) -> ep.Tensor:
    verify(isinstance(values, list), "fairbench.wmean can only reduce lists.")
    for value in values:
        if (
            not isinstance(value, Explainable)
            or "samples" not in value.explain.branches()
        ):
            raise ExplainableError("Explanation absent or does not store `samples`")
    # print([(value, value.explain.samples) for value in values])  # TODO: this is an issue with jax
    nom = _sum([value * value.explain.samples for value in values])
    denom = _sum([value.explain.samples for value in values])
    return nom if denom == 0 else nom / denom
