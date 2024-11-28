from fairbench.core_v2.values import Value, Descriptor


class Comparison:
    def __init__(self, name, description=None):
        self.descriptor = Descriptor(name, "comparison", description)
        self.depends = list()

    def instance(self, name, report: Value):
        instance_descriptor = Descriptor(name, "instance")
        self.depends.append(report.rebase(instance_descriptor))
        return self

    def clear(self):
        self.depends = list()

    def build(self):
        ret = Value(descriptor=self.descriptor, depends=self.depends)
        self.clear()
        return ret
