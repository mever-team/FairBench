from fairbench.v2.core import Descriptor, Value


class Investigator:
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
            if number is value.value:
                return value
            return descriptor(value=number, depends=list(value.depends.values()))
        depends = [self._walk(dep) for dep in value.depends.values()]
        depends = [dep for dep in depends if dep is not None]
        return descriptor(value=number, depends=depends)

    def filter(self, value: Value) -> Value:
        return self._walk(value)
