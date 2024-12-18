import eagerpy as ep
from typing import Union
import numpy as np
import sys

__fairbench_backend = "numpy"


def setbackend(backend_name: str):
    assert backend_name in ["torch", "tensorflow", "jax", "numpy"]
    global __fairbench_backend
    __fairbench_backend = backend_name


def tobackend(value):
    global __fairbench_backend
    if value.__class__.__name__ == "Fork":
        from fairbench.v1 import Fork

        return Fork({k: tobackend(v) for k, v in value.branches().items()})
    if value.__class__.__name__ == "DotDict":
        from fairbench import DotDict

        return DotDict({k: tobackend(v) for k, v in value.items()})
    name = type(value.raw if isinstance(value, ep.Tensor) else value).__module__.split(
        "."
    )[0]
    m = sys.modules
    if isinstance(value, list):
        value = np.array(value)
    if isinstance(value, float) or isinstance(value, np.float64):
        if __fairbench_backend == "numpy":
            return ep.NumPyTensor(value)
        value = float(value)
        if __fairbench_backend == "torch":
            import torch

            value = torch.tensor(value)
        elif __fairbench_backend == "tensorflow":
            import tensorflow

            value = tensorflow.convert_to_tensor(value)
        elif __fairbench_backend == "jax":
            import jax.numpy as jnp

            value = jnp.array(value)
    elif name != __fairbench_backend:
        value = value.raw if isinstance(value, ep.Tensor) else value
        if name == "torch" and isinstance(value, m[name].Tensor):  # type: ignore
            value = value.detach().numpy()
        elif name == "tensorflow" and isinstance(value, m[name].Tensor):  # type: ignore
            value = value.numpy()
        if (name == "jax" or name == "jaxlib") and isinstance(value, m["jax"].numpy.ndarray):  # type: ignore
            value = np.array(value)
        if __fairbench_backend == "torch":
            import torch

            value = torch.tensor(value)
        elif __fairbench_backend == "tensorflow":
            import tensorflow

            value = tensorflow.convert_to_tensor(value)
        elif __fairbench_backend == "jax":
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


def astensor(value, _allow_explanation=True) -> Union["Explainable", ep.Tensor]:
    if value.__class__.__name__ == "Explainable" and not _allow_explanation:
        value = value.value
    elif value.__class__.__name__ == "Explainable":
        from fairbench.v1 import Explainable

        return Explainable(
            astensor(value.value),
            explain=value.explain,
            desc=value.desc,
            units=value.units,
        )
    if isinstance(value, int) or isinstance(value, float):
        value = np.float64(value)
    if (
        "tensor" not in value.__class__.__name__.lower()
        and "array" not in value.__class__.__name__.lower()
        and not isinstance(value, np.float64)
        and not isinstance(value, np.float32)
        and not isinstance(value, list)
    ):
        return value
    if isinstance(value, list):
        value = np.array(value, dtype=np.float64)
    # if isinstance(value, np.float64):
    #    value = float(value)
    if isinstance(value, np.float32):
        value = np.array(value, np.float64)  # eagerpy can't handle float32
    value = tobackend(value)
    if value.ndim != 0:
        value = value.flatten()
    return value.float64()


def asprimitive(value, _allow_explanation=True):
    if value.__class__.__name__ == "Explainable" and not _allow_explanation:
        value = value.value
    elif value.__class__.__name__ == "Explainable":
        from fairbench.v1 import Explainable

        return Explainable(
            asprimitive(value.value),
            explain=value.explain,
            desc=value.desc,
            units=value.units,
        )
    # TODO: maybe applying this as a wrapper to methods can be faster
    if isinstance(value, ep.Tensor):
        return value.raw

    return value
