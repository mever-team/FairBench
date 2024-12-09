from typing import Iterable
from makefun import wraps
from fairbench.v2.core import (
    Descriptor,
    Value,
    Number,
    TargetedNumber,
    NotComputable,
)


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


def reduction(description, autounits=True):
    """
    Reduction mechanisms take as input an iterable of values.
    Each of those values if flattened to a list of number values.
    A wrapped method runs to aggregate the list of numbers into one number.
    The resulting numbers are all packed as the dependencies of one big value.
    """

    def strategy(func):
        descriptor = Descriptor(func.__name__, "reduction", description)

        @wraps(func)
        def wrapper(values: Iterable[Value] | Value, **kwargs) -> Value:
            if isinstance(values, Value):
                assert values.value is None, (
                    f"It is ambiguous to apply the reduction {func.__name__} on '{values.descriptor}' "
                    f"because the latter is associated with a numeric value {float(values.value):.3f}. "
                    "Maybe you meant to apply the reduction on its input's `.details` ?"
                )
                values = values.depends.values()
            values = list(values)
            ret = list()
            for arg in values:
                dependencies = list(arg.depends.values())
                flattened_arg = arg.flatten(to_float=False)

                # TODO: there is a good chance that we may want to check for the same roles too
                # find common units
                units = {
                    value.value.units
                    for value in flattened_arg
                    if value.value is not None and hasattr(value.value, "units")
                } - {None, ""}
                assert len(units) <= 1, (
                    f"More than one units were provided to reduction {func.__name__}: {', '.join(units)}. "
                    "Maybe a different view obtained with `.explain' is more suitable?"
                )

                # set descriptor
                descriptors = Descriptor(
                    descriptor.name + " " + arg.descriptor.name,
                    descriptor.role + " " + arg.descriptor.role,
                    descriptor.details + " of " + arg.descriptor.details,
                    arg.descriptor.alias,
                    prototype=arg.descriptor,
                    preferred_units=(
                        func.__name__ + " " + str(next(units.__iter__()))
                        if autounits and units
                        else None
                    ),
                )

                try:
                    value = func(flattened_arg, **kwargs)
                except NotComputable:
                    continue
                if not isinstance(value, TargetedNumber):
                    value = Number(value, units=descriptors.preferred_units)
                ret.append(descriptors(value, dependencies))
            return descriptor(depends=ret)

        wrapper.descriptor = descriptor
        return wrapper

    return strategy