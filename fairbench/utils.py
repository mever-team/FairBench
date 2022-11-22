import pyfop as pfp
import numpy
from pyfop.execution import PendingCall
from pyfop.aspect import Aspect, Priority
from inspect import signature, Parameter
from makefun import wraps


SHARED = "__FAIRBENCH_SHARED_VARIABLE__"
GENERATOR = "__FAIRBENCH_GENERATOR_VARIABLE__"


def _autoaspects(method):
    if isinstance(method, type):
        return type(method.__name__, (method,), {"__init__": _autoaspects(method.__init__)})

    params = signature(method)
    new_params = list()
    for value in params.parameters.values():
        if (hasattr(value.default, "__name__") and value.default.__name__ == '_empty') or id(value.default) == id(GENERATOR):
            new_params.append(Parameter(value.name, value.kind, default=value.default))
        else:
            new_params.append(Parameter(value.name, value.kind,
                              default=Aspect(value.default, Priority.NORMAL)
                              if not isinstance(value.default, Aspect) and not isinstance(value.default, PendingCall)
                              else value.default))

    @wraps(method, new_sig=params.replace(parameters=new_params))
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)
    return wrapper


def _autogenerator(method):
    if isinstance(method, type):
        return type(method.__name__, (method,), {"__init__": _autogenerator(method.__init__)})

    params = signature(method)
    generators_list = list()
    generators_dict = dict()
    for value in params.parameters.values():
        generators_list.append(id(value.default) == id(GENERATOR))
        generators_dict[value.name] = id(value.default) == id(GENERATOR)
    @wraps(method)
    def wrapper(*args, **kwargs):
        args = [pfp.Generator(arg) if generators_list[i] else arg for i, arg in enumerate(args)]
        kwargs = {key: pfp.Generator(val) if generators_dict[key] else val for key, val in kwargs.items()}
        return method(*args, **kwargs)
    return wrapper


def framework(method):
    return _autogenerator(pfp.lazy(_autoaspects(method)))


def instance(constructor, *args, **kwargs):
    return _autogenerator(pfp.lazy(_autoaspects(constructor)))(*args, **kwargs)


def missing(var, **kwargs):
    return [property for property, value in var.get_input_context().values.items() if id(value) == id(SHARED) and property not in kwargs]


@framework
def fit(classifier, features=SHARED, ground_truth=SHARED, sample_weight=None):
    return classifier.fit(features, ground_truth, sample_weight=sample_weight)


@framework
def array(data, backend=numpy):
    return backend.array(data, copy=False)

