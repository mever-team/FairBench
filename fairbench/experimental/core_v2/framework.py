from typing import Iterable
from makefun import wraps
from fairbench.experimental.core_v2 import Descriptor, Value, TargetedNumber


def measure(description):
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
                or isinstance(value.value, float)
                or isinstance(value.value, int)
            ), f"{descriptor} computed {type(value.value)} instead of float, int, or TargetedNumber"
            assert (
                0 <= float(value) <= 1
            ), f"{descriptor} computed {float(value)} that is not in [0,1]"
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
        def wrapper(values: Iterable[Value]) -> Value:
            values = list(values)
            ret = list()
            for arg in values:
                descriptors = Descriptor(
                    descriptor.name + " " + arg.descriptor.name,
                    descriptor.role + " " + arg.descriptor.role,
                    descriptor.details + " of " + arg.descriptor.details,
                    arg.descriptor.alias,
                    prototype=arg.descriptor
                )
                dependencies = list(arg.depends.values())
                arg = arg.flatten(to_float=False)
                value = func(arg)
                ret.append(descriptors(value, dependencies))
            return descriptor(depends=ret)

        wrapper.descriptor = descriptor
        return wrapper

    return strategy
