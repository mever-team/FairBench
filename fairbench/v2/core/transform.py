from typing import Iterable
from fairbench.v2.core import Value, NotComputable


def number(values: Iterable[Value]) -> list[float]:
    return [float(value) for value in values]


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
        min(i, j) / max(i, j) if i != 0 and j != 0 else 1
        for i in values
        for j in compared_to
    ]


def relative(
    values: Iterable[Value], compared_to: Iterable[Value] | None = None
) -> list[float]:
    values = number(values)
    compared_to = values if compared_to is None else number(compared_to)
    return [
        abs(1 - min(i, j) / max(i, j)) if i != 0 and j != 0 else 0
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
