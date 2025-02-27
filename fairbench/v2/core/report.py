from fairbench.v2.core import Sensitive, DataError, NotComputable, Descriptor
from fairbench.v1 import core as deprecated
from typing import Iterable


def report(
    sensitive: Sensitive | deprecated.Fork,
    measures: Iterable,
    reductions: Iterable,
    attach_branches_to_measures: bool = False,
    **kwargs,
):
    # prepare the sensitive attribute
    if isinstance(sensitive, dict):
        sensitive = deprecated.Fork(sensitive)
    if isinstance(sensitive, deprecated.Fork):
        sensitive = Sensitive({k: v.numpy() for k, v in sensitive.branches().items()})
    assert isinstance(
        sensitive, Sensitive
    ), "The sensitive attribute can only be a dict, Sensitive, or Fork. For example, provide `fb.categories@iterable`."

    # convert forks to dicts
    kwargs = {
        name: deprecated.Fork(arg) if isinstance(arg, dict) else arg
        for name, arg in kwargs.items()
    }
    kwargs = {
        name: (
            {k: v.raw if hasattr(v, "raw") else v for k, v in arg.branches().items()}
            if isinstance(arg, deprecated.Fork)
            else arg
        )
        for name, arg in kwargs.items()
    }

    gathered_branches = {
        branch_name
        for arg in kwargs.values()
        if isinstance(arg, dict)
        for branch_name in arg
        if branch_name not in sensitive.branches
    }
    if gathered_branches and not attach_branches_to_measures:
        # make the computations for each branch combination
        branch_reports = list()
        for branch_name in gathered_branches:
            branch_sensitive = Sensitive(
                sensitive.branches,
                Descriptor(
                    name=branch_name,
                    alias=branch_name,
                    role="branch",
                    details=f"branch {branch_name}",
                ),
            )
            # for the branch name, specialize each kwarg if the latter is a fork with that value
            branch_kwargs = dict()
            specialized_keys = list()
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    assert branch_name in v, (
                        f"Analysis argument '{k}' is missing branch '{branch_name}' that is present "
                        f"in at least one other input argument passed to the report "
                        "(the conflict is only between Fork inputs with multiple values). Consider "
                        "creating two reports, pruning branches of other values, or adding such a branch."
                    )
                    branch_kwargs[k] = v[branch_name]
                    specialized_keys.append(k)
                else:
                    branch_kwargs[k] = v
            branch_report = report(
                sensitive=branch_sensitive,
                measures=measures,
                reductions=reductions,
                **branch_kwargs,
            )
            branch_reports.append(branch_report)
        return sensitive.descriptor(depends=branch_reports)

    # make the actual computation
    try:
        results = sensitive.assessment(measures, **kwargs)
        reduction_results = list()
        for reduction in reductions:
            try:
                value = reduction(
                    results | measure for measure in results.keys("measure")
                )
                reduction_results.append(value)
            except NotComputable:
                pass
        return sensitive.descriptor(depends=reduction_results)
    except DataError as e:
        raise DataError(str(e)) from None
    except AssertionError as e:
        raise ValueError(str(e)) from None
    except ValueError as e:
        raise ValueError(str(e)) from None
    except TypeError as e:
        raise ValueError(str(e)) from None
