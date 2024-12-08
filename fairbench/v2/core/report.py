from fairbench.v2.core import Sensitive, DataError, NotComputable
from fairbench.v1 import core as deprecated
from typing import Iterable


def report(
    sensitive: Sensitive | deprecated.Fork,
    measures: Iterable,
    reductions: Iterable,
    **kwargs,
):
    # prepare the sensitive attribute
    if isinstance(sensitive, dict):
        sensitive = deprecated.Fork(sensitive)
    if isinstance(sensitive, deprecated.Fork):
        sensitive = Sensitive({k: v.numpy() for k, v in sensitive.branches().items()})
    assert isinstance(
        sensitive, Sensitive
    ), "The sensitive attribute can only be a dict, Sensitive, or Fork. For example, provide `fb.categories@iterable`."

    # convert forks to dicts
    kwargs = {
        name: deprecated.Fork(arg) if isinstance(arg, dict) else arg
        for name, arg in kwargs.items()
    }
    kwargs = {
        name: (
            {k: v.raw for k, v in arg.branches().items()}
            if isinstance(arg, deprecated.Fork)
            else arg
        )
        for name, arg in kwargs.items()
    }

    # make the actual computation
    try:
        results = sensitive.assessment(measures, **kwargs)
        reduction_results = list()
        for reduction in reductions:
            try:
                value = reduction(
                    results | measure for measure in results.keys("measure")
                )
                reduction_results.append(value)
            except NotComputable:
                pass
        return sensitive.descriptor(depends=reduction_results)
    except DataError as e:
        raise DataError(str(e)) from None
    except AssertionError as e:
        raise ValueError(str(e)) from None
    except ValueError as e:
        raise ValueError(str(e)) from None
    except TypeError as e:
        raise ValueError(str(e)) from None
