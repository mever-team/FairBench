from makefun import wraps
import eagerpy as ep
import numpy as np
import inspect
import sys
from collections.abc import Mapping

_backend = "numpy"


def setbackend(backend_name: str):
    assert backend_name in ["torch", "tensorflow", "jax", "numpy"]
    global _backend
    _backend = backend_name


def tobackend(value):
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


def astensor(value, _allow_explanation=False) -> ep.Tensor:
    if value.__class__.__name__ == "Explainable" and not _allow_explanation:
        value = value.value
    if (
        "tensor" not in value.__class__.__name__.lower()
        and "array" not in value.__class__.__name__.lower()
    ):
        return value
    if isinstance(value, list):
        value = np.array(value, dtype=np.float)
    value = tobackend(value)
    if value.ndim != 0:
        value = value.flatten()
    return value.float64()


def fromtensor(value):
    # TODO: maybe applying this as a wrapper to methods instead of submitting to dask can be faster
    if isinstance(value, ep.Tensor):
        return value.raw

    return value


class Fork(Mapping):
    def __init__(self, *args, _prefix=True, **branches):
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
                    self._branches[k + "=" + k2 if _prefix else k2] = v2
            else:
                self._branches[k] = v

    def __getattribute__(self, name):
        if name in ["_branches", "_repr_html_"] or name in dir(Fork):
            return object.__getattribute__(self, name)
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._branches:
            ret = self._branches[name]
            if ret.__class__.__name__ == "Future":
                ret = ret.result()
            return ret

        def method(*args, **kwargs):
            return call(self, name, *args, **kwargs)

        return method

    def branches(self, branch_names=None, zero_mask=False):
        return {
            branch: (value.result() if value.__class__.__name__ == "Future" else value)
            if branch_names is None or not zero_mask or branch in branch_names
            else (value.result() if value.__class__.__name__ == "Future" else value) * 0
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
        return Fork(branches | new_branches)

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
                assert len(v_keys-keys) == 0
                assert len(keys-v_keys) == 0
        return len(keys)

    def __iter__(self):
        keys = None
        for k, v in self.branches().items():
            assert isinstance(v, dict)
            v_keys = set(v.keys())
            if keys is None:
                keys = v_keys
            else:
                assert len(v_keys-keys) == 0
                assert len(keys-v_keys) == 0
        return keys.__iter__()

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

    def __or__(self, other):
        return call(self, "__or__", other)

    def __ror__(self, other):
        return call(self, "__ror__", other)

    def __call__(self, *args, **kwargs):
        return Fork(
            **{
                branch: value(*args, **kwargs)
                for branch, value in self._branches.items()
            }
        )

    def __str__(self):
        return "\n".join(
            k + ": " + str(fromtensor(v)) for k, v in self.branches().items()
        )

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


def multibranch_tensors(method):
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
            raise Exception(
                f"Method {method} annotated as @multibranch_tensors and requires at least one Fork input"
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
        return method(*args, **kwargs)

    return wrapper


def combine(*args):
    ret = {}
    for arg in args:
        assert isinstance(arg, Fork)
        ret |= arg._branches
    return Fork(ret)


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
