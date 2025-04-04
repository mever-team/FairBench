from fairbench.v2.core import Descriptor
import numpy as np
import inspect


class NotComputable(Exception):
    """THis class corresponds to a soft computational error that stops control flow but can be ignored when caught."""

    def __init__(self, description="Not computable"):
        super().__init__(description)


class DataError(Exception):
    def __init__(self, description):
        super().__init__(description)


multidimensional = Descriptor(
    "multidim", "analysis", "analysis that compares several groups."
)


class Sensitive:
    def __init__(self, branches, multidimensional=multidimensional):
        self.descriptors = {
            key: Descriptor(key, "group", "the value for group '" + key + "'.")
            for key in branches
        }
        self.branches = {key: np.array(value) for key, value in branches.items()}
        self.descriptor = multidimensional

    def keys(self):
        return self.branches.keys()

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.descriptors[item]
        item = item.descriptor
        return self.branches[item.alias]

    def assessment(self, measures, **kwargs):
        assessment_values = list()

        for key, sensitive in self.branches.items():
            descriptor = self.descriptors[key]
            measure_values = list()
            for measure in measures:
                try:
                    sig = inspect.signature(measure)
                    valid_params = set(sig.parameters)
                    valid_kwargs = {
                        k: v for k, v in kwargs.items() if k in valid_params
                    }
                    # gather all kwarg branches that are different from the sensitive attribute's branches
                    gathered_branches = {
                        branch_name
                        for arg in valid_kwargs.values()
                        if isinstance(arg, dict)
                        for branch_name in arg
                        if branch_name not in self.branches
                    }
                    if gathered_branches:
                        # make the computations for each branch combination
                        for branch_name in gathered_branches:
                            # for the branch name, specialize each kwarg if the latter is a fork with that value
                            branch_kwargs = dict()
                            specialized_keys = list()
                            for k, v in valid_kwargs.items():
                                if isinstance(v, dict):
                                    assert branch_name in v, (
                                        f"Analysis argument '{k}' is missing branch '{branch_name}' that is present "
                                        f"in at least one other input argument passed to '{measure.descriptor}' "
                                        "(the conflict is only between Fork inputs with multiple values). "
                                        f"This measure's input arguments are: {', '.join(valid_kwargs.keys())}"
                                    )
                                    branch_kwargs[k] = v[branch_name]
                                    specialized_keys.append(k)
                                else:
                                    branch_kwargs[k] = v
                            # print(branch_kwargs)
                            result = measure(**branch_kwargs, sensitive=sensitive)
                            result.descriptor = Descriptor(
                                result.descriptor.name + branch_name,
                                result.descriptor.role,
                                details=f"{result.descriptor.details} for "
                                f"{' and '.join(specialized_keys)} being {branch_name}",
                                alias=result.descriptor.alias + branch_name,
                            )
                            measure_values.append(result)
                    else:
                        # this is what would normally happen if only the sensitive attribute has branches
                        result = measure(**valid_kwargs, sensitive=sensitive)
                        measure_values.append(result)
                except AssertionError as e:
                    raise AssertionError(
                        str(e)
                        + f" while computing {measure.descriptor} for dimension `{key}`"
                    )
                except NotComputable:
                    pass
                except TypeError:
                    pass
            if measure_values:
                assessment_values.append(descriptor(depends=measure_values))
        return self.descriptor(depends=assessment_values)

    """
    def specialize(self, other: dict, name: str = ""):
        new_branches = dict()

        for branch_name, branch_value in self.branches.items():
            for other_name, other_value in other.items():
                assert isinstance(branch_value, np.ndarray)
                assert isinstance(other_value, np.ndarray)
                assert len(branch_value) == len(other_value), (
                    f"During specialization, a mismatched number of elements was found between "
                    f"{branch_name} ({len(branch_value)}) and {other_name} ({len(other_value)})"
                )
                combined_value = branch_value * other_value
                new_name = f"{branch_name}&{name}{other_name}"

                # skip empty specialization
                if np.abs(combined_value).sum() == 0:
                    continue

                # if np.abs(combined_value)

                # assert (
                #    np.abs(combined_value).sum() != 0
                # ), f"Specialization {new_name} has no elements"
                new_branches[new_name] = combined_value
        new_other = 0
        for other_value in other.values():
            new_other = other_value + new_other
        return Sensitive(new_branches, self.descriptor), new_other
    """
