from fairbench.forks.fork import Fork, astensor, comparator
from fairbench.forks.explanation import Explainable
from typing import Optional

# from fairbench.reports.accumulate import todict as tokwargs


def areduce(fork: Fork, reducer, expand=None, transform=None, branches=None):
    return reduce(fork, reducer, expand, transform, branches, name=None)


@comparator
def reduce(
    fork: Fork,
    reducer,
    expand=None,
    transform=None,
    branches=None,
    name: Optional[str] = "",
):
    if name == "":
        name = reducer.__name__
        if expand is not None:
            name += expand.__name__
        if transform is not None:
            name += transform.__name__
        if branches is not None:
            name += "[" + ",".join(branches) + "]"
    fields = None
    for branch, v in fork.branches().items():
        if branches is not None and branch not in branches:
            continue
        if fields is None:
            fields = {f: list() for f in v} if isinstance(v, dict) else list()
        if isinstance(v, dict):
            for f in v:
                fields[f].append(
                    astensor(v[f]) if transform is None else transform(astensor(v[f]))
                )
        else:
            fields.append(
                astensor(v) if transform is None else transform(astensor(v[v]))
            )
    if expand is not None:
        fields = (
            {k: expand(v) for k, v in fields.items()}
            if isinstance(fields, dict)
            else expand(fields)
        )
    result = (
        {k: Explainable(reducer(v), fork[k], desc=name) for k, v in fields.items()}
        if isinstance(fields, dict)
        else Explainable(reducer(fields), fork, desc=name)
    )
    return result if name is None else Fork({name: result})
