import pyfop as pfp
import numpy
from pyfop.execution import PendingCall, _isfop
from pyfop.aspect import Aspect, Priority
from inspect import signature, Parameter
from makefun import wraps


SHARED = "__MISSING_FAIRBENCH_SHARED_VARIABLE__"
GENERATOR = "__MISSING_FAIRBENCH_GENERATOR_VARIABLE__"


class Modal:
    def __init__(self, **modes):
        self.modes = modes

    def aspects(self, **kwargs):
        return Modal(**{mode: value.aspects(**kwargs) for mode, value in self.modes.items()})

    def __getattribute__(self, name):
        if name in ["modes"] or name in dir(Modal):
            return object.__getattribute__(self, name)
        return self.modes[name]

    def __call__(self, *args, **kwargs):
        return Modal(**{mode: value(*args, **kwargs) for mode, value in self.modes.items()})


def multimodal(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        modes = set([mode for arg in list(args)+list(kwargs.values()) if isinstance(arg, Modal) for mode in arg.modes])
        if not modes:
            return method(*args, **kwargs)
        args = (arg if isinstance(arg, Modal) else Modal(**{mode: arg for mode in modes}) for arg in args)
        kwargs = {key: arg if isinstance(arg, Modal) else Modal(**{mode: arg for mode in modes}) for key, arg in kwargs.items()}
        return Modal(**{mode: method(*(arg.modes[mode] for arg in args),
                                     **{key: arg.modes[mode] for key, arg in kwargs.items()})
                        for mode in modes})
    return wrapper


def framework(method):
    return multimodal(pfp.lazy(pfp.autoaspects(method)))


def instance(constructor, *args, **kwargs):
    return multimodal(pfp.lazy((pfp.autoaspects(constructor))))(*args, **kwargs)


def missing(var, **kwargs):
    return [property for property, value in var.get_input_context().values.items()
            if id(value) == id(SHARED) and property not in kwargs]


@framework
def fit(classifier, x, y, sample_weight=None):
    return classifier.fit(x, y, sample_weight=sample_weight)


@framework
def predict(classifier, x):
    return classifier.predict(x)


@framework
def predict_proba(classifier, x):
    ret = classifier.predict_proba(x)[:, 1]
    return ret


def aggregate(aggregator=sum, **modes):
    def _objective(**kwargs):
        values = list()
        for mode, value in modes.items():
            kwargs_local = {key: value.modes[mode] if isinstance(value, Modal) else value for key, value in kwargs.items()}
            values.append(value.modes[mode](**kwargs_local) if isinstance(value, Modal) else value(**kwargs_local))
        return aggregator(values)
    return _objective


@framework
def array(data, backend=numpy):
    return backend.array(data, copy=False)
