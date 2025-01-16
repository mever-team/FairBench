from fairbench.v2.core import Value, Descriptor


class Progress:
    def __init__(self, name: Value | str, description=None):
        if isinstance(name, Value):
            assert description is None
            assert name.value is None
            name: Value = name
            self.descriptor = name.descriptor
            self.depends = list(name.depends.values())
            return
        assert isinstance(name, str), (
            "Can only have a Value or str as the first Progress constructor argument "
            f"but `{name.__class__.__name__}` was given."
        )
        if description is None:
            description = "tracking progress across " + name
        self.descriptor = Descriptor(name, "progress", description)
        self.depends = list()

    def __setitem__(self, name, report):
        return self.instance(name, report)

    def instance(self, name, report: Value):
        assert isinstance(name, str), "Progress instances should have string names"
        assert isinstance(report, Value), "Invalid progress instance"
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

    @property
    def status(self):
        return Value(descriptor=self.descriptor, depends=self.depends)
