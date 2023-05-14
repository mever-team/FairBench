from typing import Any
import eagerpy as ep


def tofloat(value):
    if isinstance(value, ep.Tensor):
        return float(value.raw)
    return float(value)


class Explainable:
    def __init__(self, value, explain: Any = None, desc: str = None, **kwargs):
        from fairbench.forks import Fork

        self.value = value
        self.explain = Fork(kwargs) if explain is None else explain
        self.desc = desc
        if (
            not isinstance(value, float)
            and not isinstance(value, int)
            and "tensor" not in value.__class__.__name__.lower()
            and "array" not in value.__class__.__name__
        ):
            raise Exception("Can not set non-numeric as explainable", value)
        if explain is not None and kwargs:
            raise Exception("Cannot create explainable with both kwargs and a Fork")

    def __float__(self):
        return tofloat(self.value)

    def numpy(self):
        return self.value.numpy()

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()

    def sum(self):
        return self.value.sum()

    def __sub__(self, other):
        if isinstance(other, Explainable):
            other = other.value
        return self.value - other

    def __mul__(self, other):
        if isinstance(other, Explainable):
            other = other.value
        return self.value * other

    def __rmul__(self, other):
        if isinstance(other, Explainable):
            other = other.value
        return other * self.value

    def __add__(self, other):
        if isinstance(other, Explainable):
            other = other.value
        return self.value + other

    def __radd__(self, other):
        if isinstance(other, Explainable):
            other = other.value
        return other + self.value
