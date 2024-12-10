from fairbench.v2.core import Value, Curve, TargetedNumber, Descriptor


class Walker:
    def __init__(self, shallow):
        self.shallow = shallow

    def _descriptor(self, descriptor: Descriptor) -> Descriptor:
        return descriptor

    def _walk(self, value: Value) -> Value | None:
        assert isinstance(
            value, Value
        ), f"Malformed Value was provided to {self.__class__.__name__}"
        number = None if value.value is None else self._contents(value.value)
        descriptor = self._descriptor(value.descriptor)

        if self.shallow and value.value is not None:
            if number is None:
                return None
            return descriptor(value=number, depends=list(value.depends.values()))
        depends = [self._walk(dep) for dep in value.depends.values()]
        depends = [dep for dep in depends if dep is not None]
        return descriptor(value=number, depends=depends)

    def filter(self, value: Value) -> Value:
        return self._walk(value)


class Threshold(Walker):
    def __init__(self, limit, shallow=True):
        super().__init__(shallow=shallow)
        self.limit = limit

    def _contents(self, value: any) -> any:
        assert not isinstance(
            value, Value
        ), "A Value cannot have another Value as its Value but only as a dependency"  # common error
        if isinstance(value, TargetedNumber) and self.limit < abs(
            value.value - value.target
        ):
            return value
        return None
