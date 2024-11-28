from typing import Iterable
from fairbench.core_v2.values import Value


def number(values: Iterable[Value]) -> list[float]:
    return [float(value) for value in values]


def diff(values: Iterable[Value]) -> list[float]:
    values = number(values)
    return [i - j for i in values for j in values]


def ratio(values: Iterable[Value]) -> list[float]:
    values = number(values)
    # assert all(value!=0 for value in values), "Cannot compute ratio with zero values"
    return [i / j for i in values for j in values]


def relative(values: Iterable[Value]) -> list[float]:
    values = number(values)
    return [abs(1 - i / j) for i in values for j in values if i <= j != 0]
