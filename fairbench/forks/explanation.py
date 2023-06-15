from typing import Any
import eagerpy as ep
from objwrap import ClosedWrapper
import numpy as np


def tofloat(value):
    if isinstance(value, ep.Tensor):
        return float(value.raw)
    return float(value)


class ExplainableError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.explain = message
        self.value = float("NaN")
        self.distribution = None
        self.desc = None

    def __float__(self):
        return self.value

    def __int__(self):
        return self.value

    def __str__(self):
        return "---"


class Explainable(ClosedWrapper):
    def __init__(self, value, explain: Any = None, desc: str = None, **kwargs):
        from fairbench.forks import Fork

        if value.__class__.__name__ == "Future":
            value = value.result()
        if isinstance(value, int) or isinstance(value, float):
            value = np.float64(value)
        assert (
            isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, np.floating)
            or "tensor" in value.__class__.__name__.lower()
            or "array" in value.__class__.__name__
        ), f"Can not set data type as explainable: {type(value)}"
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

    def __after__(self, obj):
        return obj
