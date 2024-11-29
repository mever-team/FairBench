from fairbench.core_v2.values import Value, Descriptor


class Comparison:
    def __init__(self, name, description=None):
        self.descriptor = Descriptor(name, "comparison", description)
        self.depends = list()

    def instance(self, name, report: Value):
        instance_descriptor = Descriptor(
            name + " " + report.descriptor.name,
            report.descriptor.role + " instance",
            details=report.descriptor.details,
        )
        report = report.rebase(instance_descriptor)
        # report = instance_descriptor(depends=[report])
        self.depends.append(report)
        return self

    def clear(self):
        self.depends = list()

    def build(self):
        ret = Value(descriptor=self.descriptor, depends=self.depends)
        self.clear()
        return ret
