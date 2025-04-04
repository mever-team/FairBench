from accelerate.commands.config.default import description

from fairbench.v2.core import Value, TargetedNumber, Descriptor
from fairbench.v2.investigate.investigator import Investigator


class BL(Investigator):
    def __init__(self, encounter, prune=True, shallow=True):
        action = "keep" if prune else "colorize"
        super().__init__(shallow=shallow or action == "keep")
        assert (
            0 <= encounter <= 1
        ), "The protection applied on encountering groups should be in the range [0,1]."
        self.encounter = encounter
        self.action = action

    def filter(self, value: Value) -> Value:
        descriptor = value.descriptor
        return (
            super()
            .filter(value)
            .rebase(
                Descriptor(
                    "BL thresholds for " + descriptor.name,
                    descriptor.role,
                    "an inferred scheme that applies basic fuzzy logic (BL) to determine thresholds of "
                    + descriptor.details
                    + "\nDifferent reduction types correspond to different BL subclasses. For this reason, "
                    f"instead of setting one common threshold on what to {self.action}, "
                    f"a thresholding strategy is applied based on the truth value {self.encounter:.3f} of "
                    "the predicate 'protect group members'. This truth value is the same value across all "
                    "groups but does not necessarily correspond to a statistical measurement of reality. Rather, "
                    "it reflects a belief provided as input. The outcome of this thresholding is either acceptance "
                    "or rejection (even slight violations are considered full rejections) of the BL fairness definition. "
                    "Not clearly accepted or rejected values are obtained if this analysis is not applicable.",
                )
            )
        )

    def _contents(self, value: any) -> any:
        assert not isinstance(
            value, Value
        ), "A Value cannot have another Value as its .value but only as a dependency."
        limit = None
        if value.units.startswith("min ") or value.units.startswith("max "):
            limit = self.encounter
        elif value.units.startswith("maxerror ") or value.units.startswith("maxdiff "):
            limit = self.encounter
        elif value.units.startswith("maxrel "):
            limit = self.encounter / 2

        if limit is None and value.value is None:
            return value
        if limit is None:
            return None
        if not isinstance(value, TargetedNumber) and self.action == "keep":
            return None

        if isinstance(value, TargetedNumber) and limit < abs(
            value.value - value.target
        ):
            if self.action == "keep":
                return value
            val = value.value
            return TargetedNumber(
                val,
                target=val - 1,
                units=value.units,
            )
        if self.action == "keep":
            return None
        if isinstance(value, TargetedNumber):
            val = value.value
            return TargetedNumber(
                val,
                target=val,
                units=value.units,
            )
        return value
