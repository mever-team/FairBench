from fairbench.core_v2.values import Descriptor

assessment = Descriptor("assessment", "report")

class Sensitive:
    def __init__(self, branches):
        self.descriptors = {key: c.Descriptor(key, "attribute") for key in branches}
        self.branches = {self.descriptors[key]: np.array(value) for key, value in branches.items()}

    def keys(self):
        return self.branches.keys()

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.descriptors[item]
        item = item.descriptor
        return self.branches[item]

    def report(self, measures, **kwargs):
        assessment_values = list()
        for descriptor, sensitive in self.branches.items():
            measure_values = list()
            for measure in measures:
                try:
                    measure_values.append(measure(**kwargs, sensitive=sensitive))
                except Exception:
                    pass
            if measure_values:
                assessment_values.append(descriptor(depends=measure_values))
        return assessment(depends=assessment_values)