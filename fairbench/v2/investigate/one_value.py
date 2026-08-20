from fairbench.v2.core import Value, TargetedNumber, Descriptor
from fairbench.v2.investigate.investigator import Investigator
import math


class _OneValue(Investigator):
    def __init__(self):
        super().__init__(shallow=True)
        self.discovered: list[TargetedNumber] = list()

    def _contents(self, value):
        assert not isinstance(
            value, Value
        ), "A Value cannot have another Value as its .value but only as a dependency"  # common error
        if isinstance(value, TargetedNumber):
            self.discovered.append(value)
        return None


class WorstCase(_OneValue):
    def filter(self, value: Value) -> Value:
        self.discovered.clear()
        self._walk(value)
        number = TargetedNumber(
            value=max(abs(value.value - value.target) for value in self.discovered),
            target=0,
            units="worst bias",
        )
        return Value(
            value=number,
            descriptor=Descriptor(
                "worst bias",
                "summary",
                "summary of a report that shows the worst deviation from all computed measures from ideal target values.\n# Details\nValue of 0 means that all biases have been perfectly mitigated.\n# Caveats and recommendations\n • The report may not account for all quantities.\n • Fairness impossiblity theorems make it typically impossible to perfectly minimize this bias aggregate (unless specific correlated notions of fairness are considered).",
            ),
            depends=[value],
        )


class SkewIndex(_OneValue):
    def filter(self, value: Value) -> Value:
        self.discovered.clear()
        self._walk(value)
        total_value = sum(max(0, v.value) for v in self.discovered)
        total_target = sum(max(0, v.target) for v in self.discovered)
        number = TargetedNumber(
            value=sum(
                value.value
                / total_value
                * math.log(value.value * total_target / (total_value * value.target))
                for value in self.discovered
                if value.target > 0 and value.value > 0
            ),
            target=0,
            units="bias skew",
            bound=1,
        )
        return Value(
            value=number,
            descriptor=Descriptor(
                "skew index",
                "summary",
                "summary of a report that shows the KL-divergence across the whole report from ideal target values.\n# Details\nWhen computed across positive rate (pr) values, this becomes the Representation Skew Index (RSI) of the independent FairBench-genai project: https://prasannavijay.github.io/fairbench/whitepaper .\n# Caveats and recommendations\n • The report may not account for all quantities.\n • Fairness impossiblity theorems make it typically impossible to perfectly minimize this bias aggregate (unless specific correlated notions of fairness are considered).",
            ),
            depends=[value],
        )
