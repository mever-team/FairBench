from makefun import wraps
import eagerpy as ep
import numpy as np
import inspect


def astensor(value) -> ep.Tensor:
    if (
        "tensor" not in value.__class__.__name__.lower()
        and "array" not in value.__class__.__name__.lower()
    ):
        return value
    if isinstance(value, list):
        value = np.array(value, dtype=np.float)
    return ep.astensor(value).float64()


def fromtensor(value):
    # TODO: maybe applying this as a wrapper to methods instead of submitting to dask can be faster
    if isinstance(value, ep.Tensor):
        return value.raw

    return value


class Fork(object):
    def __init__(self, *args, **branches):
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError(
                    "Forks can only support dicts of branches as positional arguments"
                )
            for k, v in arg.items():
                if k in branches:
                    raise TypeError(f"Branch {k} provided multiple times")
                branches[k] = v
        self._branches = branches

    def __getattribute__(self, name):
        if name in ["_branches"] or name in dir(Fork):
            return object.__getattribute__(self, name)
        if name in self._branches:
            ret = self._branches[name]
            if ret.__class__.__name__ == "Future":
                ret = ret.result()
            return ret

        def method(*args, **kwargs):
            return call(self, name, *args, **kwargs)

        return method

    def branches(self):
        return {
            branch: value.result() if value.__class__.__name__ == "Future" else value
            for branch, value in self._branches.items()
        }

    def __getitem__(self, name):
        return call(self, "__getitem__", name)

    def __setitem__(self, name, value):
        return call(self, "__setitem__", name, value)

    def __abs__(self):
        return call(self, "__abs__")

    def __add__(self, other):
        return call(self, "__add__", other)

    def __radd__(self, other):
        return call(self, "__add__", other)

    def __sub__(self, other):
        return call(self, "__sub__", other)

    def __rsub__(self, other):
        return call(self, "__rsub__", other)

    def __mul__(self, other):
        return call(self, "__mul__", other)

    def __rmul__(self, other):
        return call(self, "__rmul__", other)

    def __truediv__(self, other):
        return call(self, "__truediv__", other)

    def __rtruediv__(self, other):
        return call(self, "__rtruediv__", other)

    def __floordiv__(self, other):
        return call(self, "__floordiv__", other)

    def __rfloordiv__(self, other):
        return call(self, "__rfloordiv__", other)

    def __call__(self, *args, **kwargs):
        return Fork(
            **{
                branch: value(*args, **kwargs)
                for branch, value in self._branches.items()
            }
        )

    def __repr__(self):
        return "\n".join(
            k + ": " + str(fromtensor(v)) for k, v in self.branches().items()
        )


class _NoClient:  # emulates dask.distributed.Client
    def submit(
        self,
        method,
        *args,
        workers=None,
        allow_other_workers=True,
        pure=False,
        **kwargs,
    ):
        assert allow_other_workers
        assert not pure
        return method(*args, **kwargs)


_client = _NoClient()


def distributed(*args, **kwargs):
    global _client
    from dask.distributed import Client

    _client = Client(*args, **kwargs)


def serial():
    global _client
    _client = _NoClient()


def parallel(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        branches = set(
            [
                branch
                for arg in list(args) + list(kwargs.values())
                if isinstance(arg, Fork)
                for branch in arg._branches
            ]
        )
        if not branches:
            return fromtensor(
                method(
                    *(astensor(arg) for arg in args),
                    **{key: astensor(arg) for key, arg in kwargs.items()},
                )
            )
        args = [
            arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
            for arg in args
        ]
        kwargs = {
            key: arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
            for key, arg in kwargs.items()
        }
        try:
            argnames = inspect.getfullargspec(method)[0]
            if "branch" not in kwargs and "branch" in argnames:
                kwargs["branch"] = None
            submitted = {
                branch: _client.submit(
                    fromtensor,
                    _client.submit(
                        method,
                        *(
                            _client.submit(
                                astensor,
                                arg._branches[branch],
                                workers=branch,
                                allow_other_workers=True,
                                pure=False,
                            )
                            for arg in args
                        ),
                        **{
                            key: branch
                            if key == "branch"
                            else _client.submit(
                                astensor,
                                arg._branches[branch],
                                workers=branch,
                                allow_other_workers=True,
                                pure=False,
                            )
                            for key, arg in kwargs.items()
                        },
                        workers=branch,
                        allow_other_workers=True,
                        pure=False,
                    ),
                    workers=branch,
                    allow_other_workers=True,
                    pure=False,
                )
                for branch in branches
            }
            submitted = {branch: value for branch, value in submitted.items()}
            return Fork(**submitted)
        except KeyError as e:
            raise KeyError(str(e) + " not provided for an input")

    return wrapper


def parallel_primitive(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        branches = set(
            [
                branch
                for arg in list(args) + list(kwargs.values())
                if isinstance(arg, Fork)
                for branch in arg._branches
            ]
        )
        if not branches:
            return method(
                *((arg) for arg in args),
                **{key: (arg) for key, arg in kwargs.items()},
            )
        args = [
            arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
            for arg in args
        ]
        kwargs = {
            key: arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
            for key, arg in kwargs.items()
        }
        try:
            argnames = inspect.getfullargspec(method)[0]
            if "branch" not in kwargs and "branch" in argnames:
                kwargs["branch"] = None
            submitted = {
                branch: _client.submit(
                    method,
                    *((arg._branches[branch]) for arg in args),
                    **{
                        key: branch if key == "branch" else (arg._branches[branch])
                        for key, arg in kwargs.items()
                    },
                    workers=branch,
                    allow_other_workers=True,
                    pure=False,
                )
                for branch in branches
            }
            submitted = {branch: value for branch, value in submitted.items()}
            return Fork(**submitted)
        except KeyError as e:
            raise KeyError(
                "One of the Modal inputs is missing a "
                + str(e)
                + " branch that other inputs have"
            )

    return wrapper


def combine(fork1: Fork, fork2: Fork):
    assert isinstance(fork1, Fork)
    assert isinstance(fork2, Fork)
    return Fork(fork1._branches | fork2._branches)


@parallel_primitive
def call(obj, method, *args, **kwargs):
    if callable(method):
        return method(obj, *args, **kwargs)
    return getattr(obj, method)(*args, **kwargs)


"""
def compare(**kwargs):
    for modal in kwargs.values():
        assert isinstance(modal, Modal)
    branches = set(
        [
            branch
            for arg in list(kwargs.values())
            if isinstance(arg, Modal)
            for branch in arg.branches
        ]
    )
    return Modal(
        **{branch: {key: kwargs[key].branches[branch] for key in kwargs} for branch in branches}
    )
"""
