from fairbench.v2.core import Value, TargetedNumber, Descriptor
from fairbench.v2.investigate.investigator import Investigator


class BiasAmplification(Investigator):
    def __init__(self, base: Value):
        super().__init__(shallow=True)
        assert isinstance(base, Value)
        self.base = base

    def _walk(self, value: Value, base: Value) -> Value | None:
        assert isinstance(
            value, Value
        ), f"Malformed Value was provided to {self.__class__.__name__}"
        number = value.value
        descriptor = value.descriptor
        base_descriptor = base.descriptor
        assert base_descriptor == descriptor, (
            "Mismatching reports to compute an amplification due to value: "
            + repr(base_descriptor)
            + " vs "
            + repr(descriptor)
        )

        result_descriptor = Descriptor(
            descriptor.name + " amplification",
            descriptor.role + " amplification",
            "amplification of " + descriptor.details,
            "amplification of " + descriptor.alias,
            preferred_units=(
                descriptor.preferred_units + "/" + descriptor.preferred_units
                if descriptor.preferred_units
                else None
            ),
        )
        if number is None:
            assert (
                base.value is None
            ), "Base report has value that is not existing in derived one: " + repr(
                base_descriptor
            )
            depends = [
                self._walk(dep_value, dep_base)
                for dep_value, dep_base in zip(
                    value.depends.values(), base.depends.values()
                )
            ]
            depends = [dep for dep in depends if dep is not None]
            return result_descriptor(value=number, depends=depends)
        assert base.value is not None, "Base report is missing value: " + repr(
            base_descriptor
        )
        if not isinstance(number, TargetedNumber):
            return None
        if not isinstance(base.value, TargetedNumber):
            return None
        assert number.target == base.target, "Mismatching targets for " + repr(
            descriptor
        )
        base_value = base.value.value
        if base_value == number.target:
            return None
        if 0 != number.target:
            return None
        number = TargetedNumber(
            abs(number.value - number.target) / abs(base_value - number.target),
            target=1,
            bound=(number.bound - number.target) / abs(base_value - number.target),
        )

        result_descriptor = Descriptor(
            descriptor.name + " comparison",
            descriptor.role + " compared",
            descriptor.details + " compared to its baseline value",
            "compared " + descriptor.alias,
            preferred_units=(
                descriptor.preferred_units + "/" + descriptor.preferred_units
                if descriptor.preferred_units
                else None
            ),
        )
        base_descriptor = Descriptor(
            base_descriptor.name + " baseline",
            base_descriptor.role,
            base_descriptor.details,
            base_descriptor.alias + " baseline",
            prototype=base_descriptor,
            preferred_units=base_descriptor.preferred_units,
        )

        return result_descriptor(
            value=number, depends=[value, base.rebase(base_descriptor)]
        )

    def filter(self, value: Value) -> Value:
        return self._walk(value, self.base)
