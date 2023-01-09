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


class Modal(object):
    def __init__(self, **modes):
        self.modes = modes

    def aspects(self, **kwargs):
        return Modal(
            **{mode: value.aspects(**kwargs) for mode, value in self.modes.items()}
        )

    def __getattribute__(self, name):
        if name in ["modes"] or name in dir(Modal):
            return object.__getattribute__(self, name)
        if name in self.modes:
            return self.modes[name]

        def method(*args, **kwargs):
            return call(self, name, *args, **kwargs)

        return method

    def __call__(self, *args, **kwargs):
        return Modal(
            **{mode: value(*args, **kwargs) for mode, value in self.modes.items()}
        )

    def __repr__(self):
        return "\n".join(k + ": " + str(v) for k, v in self.modes.items())

    def __or__(self, other):
        return concat(self, other)

    def __ror__(self, other):
        return concat(other, self)


def multimodal(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        modes = set(
            [
                mode
                for arg in list(args) + list(kwargs.values())
                if isinstance(arg, Modal)
                for mode in arg.modes
            ]
        )
        if not modes:
            return method(
                *(astensor(arg) for arg in args),
                **{key: astensor(arg) for key, arg in kwargs.items()},
            )
        args = [
            arg if isinstance(arg, Modal) else Modal(**{mode: arg for mode in modes})
            for arg in args
        ]
        kwargs = {
            key: arg
            if isinstance(arg, Modal)
            else Modal(**{mode: arg for mode in modes})
            for key, arg in kwargs.items()
        }
        try:
            argnames = inspect.getfullargspec(method)[0]
            if "mode" not in kwargs and "mode" in argnames:
                kwargs["mode"] = None
            return Modal(
                **{
                    mode: method(
                        *(astensor(arg.modes[mode]) for arg in args),
                        **{
                            key: mode if key == "mode" else astensor(arg.modes[mode])
                            for key, arg in kwargs.items()
                        },
                    )
                    for mode in modes
                }
            )
        except KeyError as e:
            raise KeyError(
                "One of the Modal inputs is missing a "
                + str(e)
                + " mode that other inputs have"
            )

    return wrapper


def multimodal_primitive(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        modes = set(
            [
                mode
                for arg in list(args) + list(kwargs.values())
                if isinstance(arg, Modal)
                for mode in arg.modes
            ]
        )
        if not modes:
            return method(
                *args,
                **kwargs,
            )
        args = [
            arg if isinstance(arg, Modal) else Modal(**{mode: arg for mode in modes})
            for arg in args
        ]
        kwargs = {
            key: arg
            if isinstance(arg, Modal)
            else Modal(**{mode: arg for mode in modes})
            for key, arg in kwargs.items()
        }
        try:
            argnames = inspect.getfullargspec(method)[0]
            if "mode" not in kwargs and "mode" in argnames:
                kwargs["mode"] = None
            return Modal(
                **{
                    mode: method(
                        *((arg.modes[mode]) for arg in args),
                        **{
                            key: mode if key == "mode" else (arg.modes[mode])
                            for key, arg in kwargs.items()
                        },
                    )
                    for mode in modes
                }
            )
        except KeyError as e:
            raise KeyError(
                "One of the Modal inputs is missing a "
                + str(e)
                + " mode that other inputs have"
            )

    return wrapper


@multimodal_primitive
def call(obj, method, *args, **kwargs):
    if callable(method):
        return method(obj, *args, **kwargs)
    return getattr(obj, method)(*args, **kwargs)


@multimodal
def concat(entry1, entry2):
    return entry1 | entry2


"""
def compare(**kwargs):
    for modal in kwargs.values():
        assert isinstance(modal, Modal)
    modes = set(
        [
            mode
            for arg in list(kwargs.values())
            if isinstance(arg, Modal)
            for mode in arg.modes
        ]
    )
    return Modal(
        **{mode: {key: kwargs[key].modes[mode] for key in kwargs} for mode in modes}
    )
"""
