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


class Fork(object):
    def __init__(self, **branches):
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
        return {branch: value.result() if value.__class__.__name__ == "Future" else value for branch, value in self._branches.items()}

    def __call__(self, *args, **kwargs):
        return Fork(
            **{branch: value(*args, **kwargs) for branch, value in self._branches.items()}
        )

    def __repr__(self):
        return "\n".join(k + ": " + str(v) for k, v in self.branches.items())

    def __or__(self, other):
        return concat(self, other)

    def __ror__(self, other):
        return concat(other, self)


class _NoClient:  # emulates dask.distributed.Client
    def submit(self, method, *args, workers=None, allow_other_workers=True, pure=False, **kwargs):
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
            return method(
                *(astensor(arg) for arg in args),
                **{key: astensor(arg) for key, arg in kwargs.items()},
            )
        args = [
            arg if isinstance(arg, Fork) else Fork(**{branch: arg for branch in branches})
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
                    *(astensor(arg._branches[branch]) for arg in args),
                    **{
                        key: branch if key == "branch" else astensor(arg._branches[branch])
                        for key, arg in kwargs.items()
                    },
                    workers=branch,
                    allow_other_workers=True,
                    pure=False
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
                *args,
                **kwargs,
            )
        args = [
            arg if isinstance(arg, Fork) else Fork(**{branch: arg for branch in branches})
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
            return Fork(
                **{
                    branch: method(
                        *((arg._branches[branch]) for arg in args),
                        **{
                            key: branch if key == "branch" else (arg._branches[branch])
                            for key, arg in kwargs.items()
                        },
                    )
                    for branch in branches
                }
            )
        except KeyError as e:
            raise KeyError(
                "One of the Modal inputs is missing a "
                + str(e)
                + " branch that other inputs have"
            )

    return wrapper


@parallel_primitive
def call(obj, method, *args, **kwargs):
    if callable(method):
        return method(obj, *args, **kwargs)
    return getattr(obj, method)(*args, **kwargs)


@parallel
def concat(entry1, entry2):
    return entry1 | entry2


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
