from typing import Iterable
from fairbench.v2.core import Value, NotComputable, Curve
import numpy as np
import math


def number(values: Iterable[Value]) -> list[float]:
    return [float(value) for value in values]


def single_role(values: Iterable[Value], role: str) -> list[any]:
    ret = list()
    for value in values:
        depends = value.values(role)
        assert (
            depends
        ), f"There were no dependencies for role '{role}'. Consider specializing more."
        assert (
            len(depends) == 1
        ), f"There were multiple dependencies for role '{role}'. Consider specializing more."
        ret.append(depends[0])
    return ret


def diff(
    values: Iterable[Value], compared_to: Iterable[Value] | None = None
) -> list[float]:
    values = number(values)
    compared_to = values if compared_to is None else number(compared_to)
    return [abs(i - j) for i in values for j in compared_to]


def ratio(
    values: Iterable[Value], compared_to: Iterable[Value] | None = None
) -> list[float]:
    values = number(values)
    compared_to = values if compared_to is None else number(compared_to)
    # assert all(value!=0 for value in compared_to), "Cannot compute ratio with zero values"
    return [
        min(i, j) / max(i, j) if i != 0 or j != 0 else 1
        for i in values
        for j in compared_to
    ]


def relative(
    values: Iterable[Value], compared_to: Iterable[Value] | None = None
) -> list[float]:
    values = number(values)
    compared_to = values if compared_to is None else number(compared_to)
    return [
        abs(1 - min(i, j) / max(i, j)) if i != 0 or j != 0 else 0
        for i in values
        for j in compared_to
    ]


def at_max_samples(values: Iterable[Value]) -> list[Value]:
    max_samples = 0
    max_sample_value = 0
    for value in values:
        samples = float(value.samples)
        if samples > max_samples:
            max_samples = samples
            max_sample_value = value
    if max_samples == 0:
        raise NotComputable()
    return [max_sample_value]


def curve_diff(
    values: Iterable[Value], compared_to: Iterable[Value] | None = None
) -> list[float]:
    try:
        values: list[Value] = single_role(values, role="curve")
        compared_to: list[Value] = (
            values if compared_to is None else single_role(compared_to, role="curve")
        )
    except AssertionError as e:
        raise NotComputable(e)
    return [curve_pair_diff(i.value, j.value) for i in values for j in compared_to]


def curve_pair_diff(
    curve1: Curve,
    curve2: Curve,
    skew=lambda x, y: y,
    comparator=lambda y1, y2: np.abs(y1 - y2),
) -> float:
    assert isinstance(curve1, Curve) and isinstance(
        curve2, Curve
    ), "Cannot compare non-curves"
    n_points = min(len(curve1.x), len(curve2.x))
    assert (
        n_points > 1
    ), "A curve was less than two points was involved in comparing curve areas"
    curve1_grid = curve1.to_grid(n_points)
    curve2_grid = curve2.to_grid(n_points)
    skewed_y1 = skew(curve1_grid.x, curve1_grid.y)
    skewed_y2 = skew(curve2_grid.x, curve2_grid.y)
    x_integral = np.mean(skew(curve1_grid.x, np.ones_like(curve1_grid.x)))
    return np.mean(comparator(skewed_y1, skewed_y2)) / x_integral
