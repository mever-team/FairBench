from makefun import wraps
import eagerpy as ep
import numpy as np
import inspect
import sys
from collections.abc import Mapping

_backend = "numpy"


def _str_foreign(v, tabs=0):
    if isinstance(v, Fork):
        v = v.branches()
    if isinstance(v, dict):
        complicated = False
        for val in v.values():
            if isinstance(val, Fork) or isinstance(val, dict):
                complicated = True
        return "\n".join(
            "   " * tabs
            + k
            + ": "
            + ("\n" if complicated else "")
            + _str_foreign(fromtensor(v), tabs + 1)
            for k, v in v.items()
        )
    if isinstance(v, float) or isinstance(v, np.float64):
        return f"{v:.3f}"
    return str(v)


class Forklike(dict):
    def __init__(self, *args, _role=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._role = _role

    def role(self, _role=None):
        return object.__getattribute__(self, "_role")

    def __getattribute__(self, name):
        if name in dir(Forklike):
            return object.__getattribute__(self, name)
        return self[name]

    def __getitem__(self, item):
        if item in self:
            return super().__getitem__(item)
        return Forklike({k: getattr(v, item) for k, v in self.items()})

    def __str__(self):
        return _str_foreign(self)


def _result(ret):
    if ret.__class__.__name__ == "Future":
        ret = ret.result()
    if isinstance(ret, dict):
        return Forklike(ret)
    return ret


def setbackend(backend_name: str):
    assert backend_name in ["torch", "tensorflow", "jax", "numpy"]
    global _backend
    _backend = backend_name


def tobackend(value):
    if isinstance(value, ep.Tensor) and isinstance(
        value.raw, np.float64
    ):  # TODO: investigate why this is needed for distributed mode
        return value
    global _backend
    name = type(value.raw if isinstance(value, ep.Tensor) else value).__module__.split(
        "."
    )[0]
    m = sys.modules
    if isinstance(value, list):
        value = np.array(value)
    if name != _backend:
        value = value.raw if isinstance(value, ep.Tensor) else value
        if name == "torch" and isinstance(value, m[name].Tensor):  # type: ignore
            value = value.detach().numpy()
        elif name == "tensorflow" and isinstance(value, m[name].Tensor):  # type: ignore
            value = value.numpy()
        if (name == "jax" or name == "jaxlib") and isinstance(value, m["jax"].numpy.ndarray):  # type: ignore
            value = np.array(value)
        if _backend == "torch":
            import torch

            value = torch.from_numpy(value)
        elif _backend == "tensorflow":
            import tensorflow

            value = tensorflow.convert_to_tensor(value)
        elif _backend == "jax":
            import jax.numpy as jnp

            value = jnp.array(value)
    return ep.astensor(value)


def istensor(value, _allow_explanation=False) -> bool:
    if value.__class__.__name__ == "Explainable" and not _allow_explanation:
        value = value.value
    if (
        "tensor" not in value.__class__.__name__.lower()
        and "array" not in value.__class__.__name__.lower()
    ):
        return False
    return True


def astensor(value, _allow_explanation=True) -> ep.Tensor:
    if value.__class__.__name__ == "Explainable" and not _allow_explanation:
        value = value.value
    elif value.__class__.__name__ == "Explainable":
        from fairbench import Explainable

        return Explainable(
            astensor(value.value), explain=value.explain, desc=value.desc
        )
    if isinstance(value, int) or isinstance(value, float):
        value = np.float64(value)
    if (
        "tensor" not in value.__class__.__name__.lower()
        and "array" not in value.__class__.__name__.lower()
        and not isinstance(value, np.float64)
        and not isinstance(value, list)
    ):
        return value
    if isinstance(value, list):
        value = np.array(value, dtype=np.float)
    if isinstance(value, np.float64):
        value = ep.NumPyTensor(value)
    else:
        value = tobackend(value)
    if value.ndim != 0:
        value = value.flatten()
    return value.float64()


def fromtensor(value, _allow_explanation=True):
    if value.__class__.__name__ == "Explainable" and not _allow_explanation:
        value = value.value
    elif value.__class__.__name__ == "Explainable":
        from fairbench import Explainable

        return Explainable(
            fromtensor(value.value), explain=value.explain, desc=value.desc
        )
    # TODO: maybe applying this as a wrapper to methods instead of submitting to dask can be faster
    if isinstance(value, ep.Tensor):
        return value.raw

    return value


class Fork(Mapping):
    def __init__(self, *args, _separator="", _role=None, **branches):
        self._role = _role
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError(
                    "Forks can only support dicts (holding branch values) as positional arguments"
                )
            for k, v in arg.items():
                if k in branches:
                    raise TypeError(f"Branch {k} provided multiple times")
                branches[k] = v
        self._branches = dict()
        for k, v in branches.items():
            if isinstance(v, dict) and v.__class__.__name__ == "Categorical":
                for k2, v2 in v.items():
                    self._branches[
                        str(k2) if _separator is None else k + _separator + str(k2)
                    ] = v2
            else:
                self._branches[k] = v

    def role(self):
        return object.__getattribute__(self, "_role")

    def __getattribute__(self, name):
        if name in ["_branches", "_repr_html_"] or name in dir(Fork):
            return object.__getattribute__(self, name)
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._branches:
            ret = self._branches[name]
            return _result(ret)

        # def method(*args, **kwargs):
        #    return call(self, name, *args, **kwargs)
        # return method

        return Fork(
            {
                k: v.__getattribute__(name)
                if isinstance(v, Fork)
                else call(v, "__getattribute__", name)
                for k, v in self._branches.items()
            }
        )

    def extract(self, *args):
        import fairbench as fb

        ret = dict()
        for arg in args:
            ret = fb.merge(ret, fb.todict(**{arg: self[arg]}))
        return ret

    def branches(self, branch_names=None, zero_mask=False):
        return {
            branch: _result(value)
            if branch_names is None or not zero_mask or branch in branch_names
            else _result(value) * 0
            for branch, value in self._branches.items()
            if branch_names is None or zero_mask or branch in branch_names
        }

    def withcomplements(self):
        # find missing branch complements
        branches = self.branches()
        new_branches = dict()
        for branch in branches:
            has_complement = False
            for branch2 in branches:
                if (
                    astensor(branches[branch]).abs()
                    - 1
                    + astensor(branches[branch2]).abs()
                ).abs().sum() == 0:
                    has_complement = True
                    break
            if not has_complement:
                new_branches[branch + "'"] = 1 - branches[branch]
        return Fork({**branches, **new_branches})

    def intersections(self):
        # get branches
        branches = self.branches()
        ids2names = dict(enumerate(branches))
        vec = [0] * len(branches)
        while True:
            # iterate through all different combinations
            vec[-1] += 1
            j = len(vec) - 1
            while j > 0 and vec[j] > 1:
                vec[j] = 0
                vec[j - 1] += 1
                j -= 1
            if j == 0 and vec[0] > 1:
                break
            candidates = [ids2names[i] for i in range(len(vec)) if vec[i] != 0]
            yield candidates

    def intersectional(self, delimiter="&"):
        # get branches
        branches = self.branches()
        new_branches = dict()
        for candidates in self.intersections():
            new_mask = 1
            for branch in candidates:
                new_mask = tobackend(branches[branch]) * new_mask
            if astensor(new_mask).abs().sum() == 0:
                continue
            new_branches[
                (delimiter.join(candidates)) if len(candidates) > 1 else candidates[0]
            ] = new_mask
        return Fork(new_branches)

    def __len__(self):
        keys = None
        for k, v in self.branches().items():
            assert isinstance(v, dict)
            v_keys = set(v.keys())
            if keys is None:
                keys = v_keys
            else:
                assert len(v_keys - keys) == 0
                assert len(keys - v_keys) == 0
        return len(keys)

    def __iter__(self):
        keys = None
        for k, v in self.branches().items():
            assert isinstance(v, dict)
            v_keys = set(v.keys())
            if keys is None:
                keys = v_keys
            else:
                assert len(v_keys - keys) == 0
                assert len(keys - v_keys) == 0
        return keys.__iter__()

    def __delitem__(self, name):
        return call(self, "__delitem__", name)

    def __getitem__(self, name):
        return call(self, "__getitem__", name)

    def __setitem__(self, name, value):
        return call(self, "__setitem__", name, value)

    def __abs__(self):
        return call(self, "__abs__")

    def __eq__(self, other):
        return call(self, "__eq__", other)

    def __lt__(self, other):
        return call(self, "__lt__", other)

    def __gt__(self, other):
        return call(self, "__gt__", other)

    def __le__(self, other):
        return call(self, "__le__", other)

    def __ge__(self, other):
        return call(self, "__ge__", other)

    def __ne__(self, other):
        return call(self, "__ne__", other)

    def __neg__(self):
        return call(self, "__neg__")

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

    def __or__(self, other):
        return call(self, "__or__", other)

    def __and__(self, other):
        return call(self, "__and__", other)

    def __ror__(self, other):
        return call(self, "__ror__", other)

    def __call__(self, *args, **kwargs):
        from fairbench import Explainable

        return Fork(
            **{
                branch: value(*args, **kwargs)
                if not isinstance(value, Explainable)
                else value
                for branch, value in self._branches.items()
            }
        )

    def __str__(self):
        return _str_foreign(self)

    def __repr__(self):
        # from IPython.display import display_html, HTML
        # display_html(HTML(self.__repr_html__()))
        return super().__repr__()

    def _repr_html_(self):
        return self.__repr_html__()

    def __repr_html__(self, override=None):
        if (
            override is not None
            and not isinstance(override, dict)
            and not isinstance(override, Fork)
        ):
            return override

        complex_contents = any(
            isinstance(v, dict)
            for k, v in (self.branches() if override is None else override).items()
        )
        if complex_contents:
            html = ""
            for k, v in (self.branches() if override is None else override).items():
                html += '<div style="display: inline-block; float: left;">'
                html += "<h3>{}</h3>".format(k)
                html += "{}".format(self.__repr_html__(v))
                html += "</div>"
            return html

        html = "<table>"
        for k, v in (self.branches() if override is None else override).items():
            html += "<tr>"
            html += "<td><strong>{}</strong></td>".format(k)
            if isinstance(v, dict):
                html += "<td>{}</td>".format(self.__repr_html__(v))
            else:
                html += "<td>{}</td>".format(fromtensor(v))
            html += "</tr>"
        html += "</table>"
        return html


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

    if isinstance(_client, Client):
        _client.close()
    _client = Client(*args, **kwargs)


def serial():
    global _client
    from dask.distributed import Client

    if isinstance(_client, Client):
        _client.close()
    _client = _NoClient()


def parallel(_wrapped_method):
    if len(inspect.getfullargspec(_wrapped_method)[0]) <= 1:
        raise Exception(
            "To avoid ambiguity, the @parallel decorator can be applied only to methods with at least"
            "two arguments."
        )

    @wraps(_wrapped_method)
    def wrapper(*args, **kwargs):
        if len(args) == 1 and not kwargs:
            argnames = inspect.getfullargspec(_wrapped_method)[0]
            arg = args[0]
            kwargs = {k: getattr(arg, k) for k in argnames if hasattr(arg, k)}
            args = []

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
                _wrapped_method(
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
            argnames = inspect.getfullargspec(_wrapped_method)[0]
            if "branch" not in kwargs and "branch" in argnames:
                kwargs["branch"] = None
            submitted = {
                branch: _client.submit(
                    fromtensor,
                    _client.submit(
                        _wrapped_method,
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


def role(rolename):
    def decorator(_wrapped_method):
        @wraps(_wrapped_method)
        def wrapper(*args, **kwargs):
            ret = _wrapped_method(*args, **kwargs)
            if isinstance(ret, Forklike) or isinstance(ret, Fork):
                object.__setattr__(ret, "_role", rolename)
            return ret
        return wrapper
    return decorator


def comparator(_wrapped_method):
    @wraps(_wrapped_method)
    def wrapper(*args, **kwargs):
        has_fork_of_forks = False
        for arg in args:
            if isinstance(arg, Fork):
                for k, v in arg._branches.items():
                    if isinstance(v, Fork):
                        has_fork_of_forks = True
        for arg in kwargs.values():
            if isinstance(arg, Fork):
                for k, v in arg._branches.items():
                    if isinstance(v, Fork):
                        has_fork_of_forks = True
        if not has_fork_of_forks:
            return _wrapped_method(*args, **kwargs)

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
                _wrapped_method(
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
            argnames = inspect.getfullargspec(_wrapped_method)[0]
            if "branch" not in kwargs and "branch" in argnames:
                kwargs["branch"] = None
            submitted = {
                branch: _client.submit(
                    fromtensor,
                    _client.submit(
                        _wrapped_method,
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


def parallel_primitive(_wrapped_method):
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
            return _wrapped_method(
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
            argnames = inspect.getfullargspec(_wrapped_method)[0]
            if "branch" not in kwargs and "branch" in argnames:
                kwargs["branch"] = None
            submitted = {
                branch: _client.submit(
                    _wrapped_method,
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


def multibranch_tensors(_wrapped_method):
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
            arg
            if isinstance(arg, Fork) or not istensor(arg, True)
            else Fork(**{branch: astensor(arg) for branch in branches})
            for arg in args
        ]
        kwargs = {
            key: arg
            if isinstance(arg, Fork) or not istensor(arg)
            else Fork(**{branch: astensor(arg) for branch in branches})
            for key, arg in kwargs.items()
        }
        return _wrapped_method(*args, **kwargs)

    return wrapper


@parallel_primitive
def merge(dict1, dict2):
    return Forklike({**dict1, **dict2})


@comparator
def combine(*args, _role=None):
    ret = {}
    for arg in args:
        assert isinstance(arg, Fork)
        ret = merge(ret, arg._branches)
    return Fork(ret)


def unit_bounded(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        for iter in [args, kwargs.values()]:
            for arg in iter:
                if isinstance(arg, ep.Tensor):
                    assert (
                        arg.min() >= 0 and arg.max() <= 1
                    ), f"{method.__name__} inputs should lie in the range [0,1]. Maybe use fairbench.categories to transform categorical data."
        return method(*args, **kwargs)

    return wrapper


@parallel_primitive
def call(obj, method, *args, **kwargs):
    if method == "__getattribute__":
        obj = _result(obj)
    if (
        method == "__getattribute__"
        and isinstance(obj, dict)
        and len(args) == 1
        and len(kwargs) == 0
    ):
        return obj[args[0]]
    if callable(method):
        return method(obj, *args, **kwargs)
    """
    def run(obj, method, *args, **kwargs):
        attr = getattr(obj, method)
        if not callable(attr):
            return attr
        return attr(*args, **kwargs)
    return _client.submit(run, obj, method, *args, **kwargs, pure=False)
    """
    attr = getattr(obj, method)
    if not callable(attr):
        return attr
    return attr(*args, **kwargs)


"""
def compare(**todict):
    for modal in todict.values():
        assert isinstance(modal, Modal)
    branches = set(
        [
            branch
            for arg in list(todict.values())
            if isinstance(arg, Modal)
            for branch in arg.branches
        ]
    )
    return Modal(
        **{branch: {key: todict[key].branches[branch] for key in todict} for branch in branches}
    )
"""
