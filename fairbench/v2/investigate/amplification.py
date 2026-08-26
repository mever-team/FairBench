from fairbench.v2.core import Value, TargetedNumber, Descriptor
from fairbench.v2.investigate.investigator import Investigator


class Amplification(Investigator):
    def __init__(self, base: Value, force=False, default_to_zero_targets=False):
        super().__init__(shallow=True)
        assert isinstance(base, Value)
        self.base = base
        self.force = force
        self.default_to_zero_targets = default_to_zero_targets

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
        target_number = None
        if target_number is None and isinstance(number, TargetedNumber):
            target_number = number.target
        if target_number is None and isinstance(base.value, TargetedNumber):
            target_number = base.value.target
        if target_number is None:
            if not self.default_to_zero_targets:
                return None
            target_number = 0.0
        assert isinstance(target_number, float | int)
        assert (
            not isinstance(base.value, TargetedNumber)
            or target_number == base.value.target
        ), "Mismatching targets for " + repr(descriptor)
        assert (
            not isinstance(number, TargetedNumber) or target_number == number.target
        ), "Mismatching targets for " + repr(descriptor)

        number = TargetedNumber(
            abs(number.value - target_number) / abs(base.value.value - target_number),
            target=1,
            bound=(number.bound - target_number)
            / abs(base.value.value - target_number),
        )
        return result_descriptor(
            value=number, depends=[value, base.rebase(base_descriptor)]
        )

    def filter(self, value: Value) -> Value:
        return self._walk(value, self.base)
