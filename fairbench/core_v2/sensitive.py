from fairbench.core_v2.values import Descriptor
import numpy as np

class NotComputable(Exception):
    def __init__(self, description="Not computable"):
        super().__init__(description)

class DataError(Exception):
    def __init__(self, description):
        super().__init__(description)

assessment = Descriptor("assessment", "results")

class Sensitive:
    def __init__(self, branches):
        self.descriptors = {key: Descriptor(key, "attribute") for key in branches}
        self.branches = {self.descriptors[key]: np.array(value) for key, value in branches.items()}

    def keys(self):
        return self.branches.keys()

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.descriptors[item]
        item = item.descriptor
        return self.branches[item]

    def assessment(self, measures, **kwargs):
        assessment_values = list()
        for descriptor, sensitive in self.branches.items():
            measure_values = list()
            for measure in measures:
                try:
                    measure_values.append(measure(**kwargs, sensitive=sensitive))
                except NotComputable:
                    pass
            if measure_values:
                assessment_values.append(descriptor(depends=measure_values))
        return assessment(depends=assessment_values)