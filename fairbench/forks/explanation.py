from typing import Any
import eagerpy as ep
from objwrap import Wrapper


def tofloat(value):
    if isinstance(value, ep.Tensor):
        return float(value.raw)
    return float(value)


class Explainable(Wrapper):
    def __init__(self, value, explain: Any = None, desc: str = None, **kwargs):
        from fairbench.forks import Fork

        if value.__class__.__name__ == "Future":
            value = value.result()

        assert (
            isinstance(value, float)
            or isinstance(value, int)
            or "tensor" in value.__class__.__name__.lower()
            or "array" in value.__class__.__name__
        ), "Can not set non-numeric as explainable"
        assert (
            explain is None or not kwargs
        ), "Cannot create explainable with both todict and a Fork"
        super().__init__(value)
        self.explain = Fork(kwargs) if explain is None else explain
        self.desc = desc

    def __float__(self):
        return tofloat(self.__value__())

    def __int__(self):
        return int(self.__float__())

    def __str__(self):
        return f"{self.__float__():.3f}"

    @property
    def value(self):
        return self.__value__()

    def numpy(self):
        return self.value.numpy()
