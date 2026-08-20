from fairbench.v2.core import Value, TargetedNumber, Number, Descriptor
from fairbench.v2.investigate.investigator import Investigator


class SetTargets(Investigator):
    def __init__(self, targets: Value, force=False):
        super().__init__(shallow=True)
        assert isinstance(targets, Value)
        self.base = targets
        self.force = force

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
        assert (
            not isinstance(number, TargetedNumber)
            or not isinstance(base.value, TargetedNumber)
            or number.target == base.target
        ), "Mismatching targets for " + repr(descriptor)
        # the following two asserts should never complain, unless things go internally wrong
        assert base.value is not None
        assert isinstance(number, TargetedNumber) or isinstance(number, Number)
        assert (
            not base.value.units or not number.units or base.value.units == number.units
        ), (
            "Mismatching units when comparing values:"
            + "\n - "
            + str(number)
            + "\n - "
            + str(base)
        )
        base_value = base.value.value
        number = TargetedNumber(
            number.value,
            target=base_value,
            bound=number.bound,
            units=(number.units if number.units else base.value.units)
            + f" (target baseline {float(base.value):.3f})",
        )
        result_descriptor = Descriptor(
            descriptor.name,
            descriptor.role,
            descriptor.details + f" must be compared to target baseline",
            "compared " + descriptor.alias,
            preferred_units=descriptor.preferred_units,
        )
        base_descriptor = Descriptor(
            "baseline",
            base_descriptor.role,
            base_descriptor.details,
            "baseline",
            prototype=base_descriptor,
            preferred_units=base_descriptor.preferred_units,
        )

        return result_descriptor(
            value=number,
            depends=[*value.depends.values(), base.rebase(base_descriptor)],
        )

    def filter(self, value: Value) -> Value:
        return self._walk(value, self.base)
