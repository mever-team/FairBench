from fairbench.v1.core.compute import (
    comparator,
    parallel_primitive,
    asprimitive,
    astensor,
    istensor,
)
import numpy as np
import eagerpy as ep
from makefun import wraps


def _result(ret):
    from fairbench.v1 import DotDict

    if ret.__class__.__name__ == "Future":
        ret = ret.result()
    if isinstance(ret, dict):
        return DotDict(ret)
    return ret


def role(rolename):
    """Sets the _role attribute of any returned Fork or DotDict."""
    from fairbench.v1.core import DotDict
    from fairbench.v1.core import Fork

    def decorator(_wrapped_method):
        @wraps(_wrapped_method)
        def wrapper(*args, **kwargs):
            ret = _wrapped_method(*args, **kwargs)
            if isinstance(ret, DotDict) or isinstance(ret, Fork):
                object.__setattr__(ret, "_role", rolename)
            return ret

        return wrapper

    return decorator


def simplify(fork):
    from fairbench.v1 import Explainable, ExplainableError, Fork, DotDict

    branches = fork.branches() if isinstance(fork, Fork) else fork
    branches = {
        k: simplify(v) if isinstance(v, Fork) or isinstance(v, DotDict) else v
        for k, v in branches.items()
        if not isinstance(v, ExplainableError)
    }
    if not branches:
        return ExplainableError("Branch holds no values.")
    return Fork(branches) if isinstance(fork, Fork) else DotDict(**branches)


def _str_foreign(v, tabs=0):
    from fairbench.v1 import Fork

    if isinstance(v, Fork):
        v = v.branches()
    if isinstance(v, dict):
        complicated = False
        for val in v.values():
            if isinstance(val, Fork) or isinstance(val, dict):
                complicated = True
        return "\n".join(
            "   " * tabs
            + k.ljust(max(30 - 2 * tabs, 0))
            + " "
            + ("\n" if complicated else "")
            + _str_foreign(asprimitive(v), tabs + 1)
            for k, v in v.items()
        )
    if isinstance(v, float) or isinstance(v, np.float64) or isinstance(v, np.float32):
        return f"{v:.3f}"
    return str(v)


def multibranch_tensors(_wrapped_method):
    from fairbench.v1.core.fork import Fork

    @wraps(_wrapped_method)
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
            raise Exception(
                f"Method {_wrapped_method} annotated as @multibranch_tensors and requires at least one Fork input"
            )
        args = [
            (
                arg
                if isinstance(arg, Fork) or not istensor(arg, True)
                else Fork(**{branch: astensor(arg) for branch in branches})
            )
            for arg in args
        ]
        kwargs = {
            key: (
                arg
                if isinstance(arg, Fork) or not istensor(arg)
                else Fork(**{branch: astensor(arg) for branch in branches})
            )
            for key, arg in kwargs.items()
        }
        return _wrapped_method(*args, **kwargs)

    return wrapper


@parallel_primitive
def merge(dict1, dict2):
    from fairbench.v1 import DotDict

    return DotDict({**dict1, **dict2})


@comparator
def combine(*args, _role=None, _cast=None):
    from fairbench.v1 import Fork, DotDict

    if _cast is None:
        _cast = Fork
    ret = {}
    for arg in args:
        assert isinstance(arg, Fork) or isinstance(arg, DotDict)
        ret = merge(ret, arg._branches if isinstance(arg, Fork) else arg)
    return _cast(ret)


def unit_bounded(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        for iter in [args, kwargs.values()]:
            for arg in iter:
                if (
                    isinstance(arg, ep.Tensor) and arg.shape
                ):  # do not check for single number parameters
                    assert (
                        arg.min() >= 0 and arg.max() <= 1
                    ), f"{method.__name__} inputs should lie in the range [0,1]. Maybe use fairbench.categories to transform categorical data."
        return method(*args, **kwargs)

    return wrapper


@parallel_primitive
def call(obj, method, *args, **kwargs):
    from fairbench.v1 import ExplainableError

    if method == "__getattribute__":
        obj = _result(obj)
    if (
        method == "__getattribute__"
        and isinstance(obj, dict)
        and len(args) == 1
        and len(kwargs) == 0
    ):
        return obj[args[0]]
    try:
        if callable(method):
            return method(obj, *args, **kwargs)
    except ExplainableError as e:
        return e.caught()
    attr = getattr(obj, method)
    if not callable(attr):
        return attr
    return attr(*args, **kwargs)
