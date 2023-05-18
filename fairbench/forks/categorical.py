from fairbench.forks.fork import tobackend, istensor
from typing import Iterable, Mapping


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


@Transform
def binary(x):
    x = tobackend(x)
    return {"1": x, "0": 1 - x}


@Transform
def categories(x):
    assert isinstance(x, Iterable)
    if isinstance(x, Mapping):
        return Categorical(x)
    vals = list(set(x))
    return {
        str(val.numpy())
        if istensor(val)
        else str(val): tobackend([1 if val == xval else 0 for xval in x])
        for val in vals
    }
