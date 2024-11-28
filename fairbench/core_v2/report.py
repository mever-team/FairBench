from sympy.liealgebras.type_e import TypeE

from fairbench.core_v2.values import Descriptor
from fairbench.core_v2.sensitive import Sensitive, DataError, NotComputable


report_descriptor = Descriptor("report", "results")


def report(sensitive: Sensitive, measures, reductions, **kwargs):
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
        return report_descriptor(depends=reduction_results)
    except DataError as e:
        raise DataError(str(e)) from None
    except AssertionError as e:
        raise ValueError(str(e)) from None
    except ValueError as e:
        raise ValueError(str(e)) from None
    except TypeError as e:
        raise ValueError(str(e)) from None
