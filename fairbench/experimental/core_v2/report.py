from fairbench.experimental.core_v2 import Sensitive, DataError, NotComputable
from fairbench import core as deprecated
from typing import Iterable


def report(
    sensitive: Sensitive | deprecated.Fork,
    measures: Iterable,
    reductions: Iterable,
    **kwargs
):
    if isinstance(sensitive, deprecated.Fork):
        sensitive = Sensitive({k: v.numpy() for k, v in sensitive.branches().items()})
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
