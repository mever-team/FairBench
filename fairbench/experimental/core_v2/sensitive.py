from fairbench.experimental.core_v2 import Descriptor
import numpy as np
import inspect


class NotComputable(Exception):
    """THis class corresponds to a soft computational error that stops control flow but can be ignored when caught."""

    def __init__(self, description="Not computable"):
        super().__init__(description)


class DataError(Exception):
    def __init__(self, description):
        super().__init__(description)


multidimensional = Descriptor("multidim", "analysis", "analysis for several groups")


class Sensitive:
    def __init__(self, branches, multidimensional=multidimensional):
        self.descriptors = {
            key: Descriptor(key, "group", "the value for group '" + key + "'")
            for key in branches
        }
        self.branches = {key: np.array(value) for key, value in branches.items()}
        self.descriptor = multidimensional

    def keys(self):
        return self.branches.keys()

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.descriptors[item]
        item = item.descriptor
        return self.branches[item.alias]

    def assessment(self, measures, **kwargs):
        assessment_values = list()
        for key, sensitive in self.branches.items():
            descriptor = self.descriptors[key]
            measure_values = list()
            for measure in measures:
                try:
                    sig = inspect.signature(measure)
                    valid_params = set(sig.parameters)
                    valid_kwargs = {
                        k: v for k, v in kwargs.items() if k in valid_params
                    }
                    measure_values.append(measure(**valid_kwargs, sensitive=sensitive))
                except NotComputable:
                    pass
                except TypeError:
                    pass
            if measure_values:
                assessment_values.append(descriptor(depends=measure_values))
        return self.descriptor(depends=assessment_values)
