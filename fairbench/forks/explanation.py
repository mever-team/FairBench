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
        if (
            not isinstance(value, float)
            and not isinstance(value, int)
            and "tensor" not in value.__class__.__name__.lower()
            and "array" not in value.__class__.__name__
        ):
            raise Exception("Can not set non-numeric as explainable", value)
        if explain is not None and kwargs:
            raise Exception("Cannot create explainable with both todict and a Fork")
        super().__init__(value)
        self.explain = Fork(kwargs) if explain is None else explain
        self.desc = desc

    def __float__(self):
        return tofloat(self.__value__())

    @property
    def value(self):
        return self.__value__()

    def numpy(self):
        return self.value.numpy()
