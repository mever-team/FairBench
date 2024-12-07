from typing import Any
from objwrap import ClosedWrapper
import numpy as np
import eagerpy as ep
from fairbench.v1.core.explanation.error import ExplainableError


def tofloat(value: Any) -> float:
    if isinstance(value, ep.Tensor):
        return float(value.raw)
    return float(value)


class Explainable(ClosedWrapper):
    def __init__(
        self, value, explain: Any = None, desc: str = None, units: Any = None, **kwargs
    ):
        from fairbench.v1.core import Fork

        if value.__class__.__name__ == "Future":
            value = value.result()
        if isinstance(value, np.bool_):
            value = bool(value)
        if isinstance(value, int) or isinstance(value, float):
            value = np.float64(value)
        assert (
            isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, np.floating)
            or "tensor" in value.__class__.__name__.lower()  # torch and tensorflow
            or "array" in value.__class__.__name__  # numpy
            or "ArrayImpl" in value.__class__.__name__  # jax
        ), f"Can not set data type as explainable: {type(value)}"
        assert (
            explain is None or not kwargs
        ), "Cannot create explainable with both todict and a Fork"
        super().__init__(value)
        self.explain = Fork(kwargs) if explain is None else explain
        self.desc = desc
        self.units = units

    def __float__(self):
        return tofloat(self.__value__())

    def __int__(self):
        return int(self.__float__())

    def __str__(self):
        value = self.__float__()
        if self.units is not None and (callable(self.units)):
            return str(self.units(value))
        if self.units is not None:
            return f"{value:.3f} {self.units}"
        return f"{value:.3f}"

    @property
    def value(self):
        return self.__value__()

    def numpy(self):
        return self.value.numpy()

    def __after__(self, obj):
        return obj

    def __wrapcall__(self, obj, name, *args, **kwargs):
        for arg in args:
            if isinstance(arg, ExplainableError):
                arg.reraise()
        return super().__wrapcall__(obj, name, *args, **kwargs)
