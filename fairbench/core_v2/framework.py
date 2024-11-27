from makefun import wraps

from fairbench.core_v2.values import Descriptor, Value


def reduction(description):
    def strategy(func):
        descriptor = Descriptor(func.__name__, "reduction", description)
        @wraps(func)
        def wrapper(values: Value):
            values_descriptor = values.descriptor
            values = values.flatten(to_float=False)
            value = func([float(value)for value in values])
            return Value(None, descriptor, [Value(value, values_descriptor, values)])
        return wrapper
    return strategy