from fairbench.v2.core import Value, TargetedNumber
from fairbench.v2.investigate.investigator import Investigator


class DeviationsOver(Investigator):
    def __init__(self, limit, prune=True, shallow=True):
        action = "keep" if prune else "colorize"
        super().__init__(shallow=shallow and action == "keep")
        self.limit = limit
        self.action = action

    def _contents(self, value: any) -> any:
        assert not isinstance(
            value, Value
        ), "A Value cannot have another Value as its .value but only as a dependency"  # common error
        if isinstance(value, TargetedNumber) and self.limit < abs(
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
