from typing import Iterable
from makefun import wraps
from fairbench.experimental.core_v2 import Descriptor, Value, Number, TargetedNumber


def measure(description, unit=True):
    """
    Measures compute a float value that is wrapped with their own descriptor
    and dependencies.
    """

    def strategy(func):
        descriptor = Descriptor(func.__name__, "measure", description)

        @wraps(func)
        def wrapper(**kwargs) -> Value:
            value = func(**kwargs)
            if not isinstance(value, Value):
                value = Value(value, descriptor, [])
            assert (
                isinstance(value.value, TargetedNumber)
                or isinstance(value.value, Number)
                or isinstance(value.value, float)
                or isinstance(value.value, int)
            ), f"{descriptor} computed {type(value.value)} instead of float, int, Number, or TargetedNumber"
            assert not unit or (
                0 <= float(value) <= 1
            ), f"{descriptor} computed {float(value)} that is not in [0,1]"
            if isinstance(value.value, Number) or isinstance(
                value.value, TargetedNumber
            ):
                if not value.value.units:
                    value.value.units = func.__name__
            return value.rebase(descriptor)

        wrapper.descriptor = descriptor
        return wrapper

    return strategy


def reduction(description):
    """
    Reduction mechanisms take as input an iterable of values.
    Each of those values if flattened to a list of number values.
    A wrapped method runs to aggregate the list of numbers into one number.
    The resulting numbers are all packed as the dependencies of one big value.
    """

    def strategy(func):
        descriptor = Descriptor(func.__name__, "reduction", description)

        @wraps(func)
        def wrapper(values: Iterable[Value], **kwargs) -> Value:
            values = list(values)
            ret = list()
            for arg in values:
                descriptors = Descriptor(
                    descriptor.name + " " + arg.descriptor.name,
                    descriptor.role + " " + arg.descriptor.role,
                    descriptor.details + " of " + arg.descriptor.details,
                    arg.descriptor.alias,
                    prototype=arg.descriptor,
                )
                dependencies = list(arg.depends.values())
                arg = arg.flatten(to_float=False)
                value = func(arg, **kwargs)
                if not isinstance(value, TargetedNumber):
                    value = Number(value)
                ret.append(descriptors(value, dependencies))
            return descriptor(depends=ret)

        wrapper.descriptor = descriptor
        return wrapper

    return strategy
