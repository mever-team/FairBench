from fairbench.v2.core import Value, TargetedNumber, Descriptor
from fairbench.v2.investigate.investigator import Investigator


class Amplification(Investigator):
    def __init__(self, base: Value, force=False, assume_zero_targets=False):
        super().__init__(shallow=True)
        assert isinstance(base, Value)
        self.base = base
        self.force = force
        self.assume_zero_targets = assume_zero_targets

    def _walk(self, value: Value, base: Value) -> Value | None:
        assert isinstance(
            value, Value
        ), f"Malformed Value was provided to {self.__class__.__name__}"
        number = value.value
        descriptor = value.descriptor
        base_descriptor = base.descriptor
        assert self.force or base_descriptor == descriptor or number is None, (
            "Mismatching reports to compute an amplification due to incompatibility (use 'SetTargets(targets=..., force=True)' if you are sure):"
            + "\n - "
            + repr(base_descriptor)
            + "\n - "
            + repr(descriptor)
        )

        result_descriptor = Descriptor(
            descriptor.name + " comparison",
            descriptor.role + " comparison",
            "comparison of " + descriptor.details,
            "comparison of " + descriptor.alias,
            preferred_units=(
                descriptor.preferred_units if descriptor.preferred_units else None
            ),
        )
        if number is None:
            assert base.value is None, (
                "Base report has value that does not exist in the filtered one: "
                + repr(base_descriptor)
            )
            assert self.force or len(value.depends) == len(base.depends), (
                "Different number of dependencies when comparing:"
                + "\n - "
                + repr(base_descriptor)
                + ",".join(repr(v) for v in value.depends)
                + "\n - "
                + repr(descriptor)
                + ": "
                + ",".join(repr(v) for v in base.depends)
            )
            depends = [
                self._walk(
                    dep_value,
                    (
                        base[dep_value.descriptor].single_entry()
                        if dep_value.value is not None
                        else base[dep_value.descriptor]
                    ),
                )
                for dep_value in value.depends.values()
            ]
            depends = [dep for dep in depends if dep is not None]
            return result_descriptor(value=number, depends=depends)
        assert (
            self.force or base.value is not None
        ), "Base report is missing value: " + repr(base_descriptor)
        assert base.value is not None, "Base report is missing value: " + repr(
            base_descriptor
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
        base_value = base.value.value
        if not isinstance(number, TargetedNumber) or not isinstance(base.value, TargetedNumber):
            if not self.assume_zero_targets: 
                return None
            number = TargetedNumber(
                number.value / base_value,
                target=1
            )
            return result_descriptor(
                value=number, depends=[value, base.rebase(base_descriptor)]
            )
        assert number.target == base.target, "Mismatching targets for " + repr(
            descriptor
        )
        if base_value == number.target:
            return None
        # if 0 != number.target:
        #     return None
        number = TargetedNumber(
            abs(number.value - number.target) / abs(base_value - number.target),
            target=1,
            bound=(number.bound - number.target) / abs(base_value - number.target),
        )

        return result_descriptor(
            value=number, depends=[value, base.rebase(base_descriptor)]
        )

    def filter(self, value: Value) -> Value:
        return self._walk(value, self.base)
