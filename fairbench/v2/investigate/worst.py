from fairbench.v2.core import Value, TargetedNumber
from fairbench.v2.investigate.investigator import Investigator
from fairbench.v2.investigate.deviations_over import DeviationsOver


class Worst(Investigator):
    def __init__(self, prune=True, shallow=True):
        action = "keep" if prune else "colorize"
        super().__init__(shallow=shallow or action == "keep")
        self.action = action
        self._values = [0]

    def _contents(self, value: Value) -> Value:
        if isinstance(value, TargetedNumber):
            self._values.append(abs(value.value - value.target))
        return value

    def filter(self, value: Value) -> Value:
        self._values = [0]
        self._walk(value)
        return DeviationsOver(max(self._values) - 0.55, self.action).filter(value)
