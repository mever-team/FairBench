from fairbench.v1.core.compute import tobackend, istensor
from fairbench.v1.core.fork import Fork
from typing import Iterable, Mapping
import numpy as np


class Categorical(dict):
    def __and__(self, other):
        ret = Categorical()
        for k, v in self.items():
            for k2, v2 in other.items():
                ret[k + "&" + k2] = v * v2
        return ret

    def __or__(self, other):
        ret = Categorical()
        for k, v in self.items():
            ret[k] = v
        for k, v in other.items():
            assert k not in ret
            ret[k] = v
        return ret

    def __repr__(self):
        return "Categorical: " + super().__repr__()


class Transform:
    def __init__(self, method):
        self._method = method

    def __matmul__(self, other):
        ret = self._method(other)
        assert isinstance(ret, dict)  # sanity check to avoid errors later on
        return Categorical(
            ret
        )  # the Categorical is unfolded by Fork constructors (native dicts are not)

    def __call__(
        self, other
    ):  # allow traditional call in case someone finds it easier to read
        return self.__matmul__(other)


def _onehot(num, position):
    ret = np.zeros((num))
    ret[position] = 1
    ret = tobackend(ret)
    return ret


@Transform
def individuals(x):
    if not isinstance(x, int):
        assert isinstance(x, Iterable)
        x = len(x)
    return {str(i): _onehot(x, i) for i in range(x)}


@Transform
def binary(x):
    x = tobackend(x)
    return {"1": x, "0": 1 - x}


class Categories:
    def __init__(
        self, values: Iterable, generator=lambda data, category: data == category
    ):
        self.categories = list(values)
        self.generator = generator

    def __call__(self, other):
        return self.__matmul__(other)

    def __matmul__(self, other):
        assert not isinstance(other, Fork)
        return Categorical(
            {
                str(category): self.generator(other, category)
                for category in self.categories
            }
        )


@Transform
def categories(x):
    assert isinstance(x, Iterable)
    if isinstance(x, Mapping):
        return Categorical(x)
    vals = list(set(x) - {np.nan})  # ignores nones
    return {
        str(val.numpy()) if istensor(val) else str(val): tobackend(
            [1 if val == xval else 0 for xval in x]
        )
        for val in vals
    }


@Transform
def fuzzy(x):
    assert isinstance(x, Iterable)
    if isinstance(x, Mapping):
        return Categorical(x)
    x = x.numpy() if istensor(x) else [xval for xval in x]
    x = tobackend(x)
    minx = x.min().raw
    maxx = x.max().raw
    if minx == maxx:
        return x * 0
    x = (x - minx) / (maxx - minx)
    return {f"large {float(maxx):.3f}": x, f"small {float(minx):.3f}": 1 - x}
