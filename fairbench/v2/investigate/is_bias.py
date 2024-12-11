from fairbench.v2.core import Value, TargetedNumber
from fairbench.v2.investigate.investigator import Investigator


class IsBias(Investigator):
    def __init__(self, shallow=True):
        super().__init__(shallow=shallow)

    def _contents(self, value: any) -> any:
        assert not isinstance(
            value, Value
        ), "A Value cannot have another Value as its .value but only as a dependency"  # common error
        if isinstance(value, TargetedNumber) and value.target == 0:
            return value
        return None
