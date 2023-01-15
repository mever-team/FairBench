from fairbench.forks.fork import parallel, parallel_primitive
import eagerpy as ep


"""
This module provides helper methods to concatenate tensors stored within Forks of tensor or Forks of dicts of tensors
and use the final output in one report at the end.
"""


@parallel_primitive
def kwargs(**kwargs):
    if not kwargs:
        return None
    return kwargs


@parallel_primitive
def concatenate(*data):
    data = [d for d in data if d is not None]
    if len(data) == 1:
        return data[0]
    isdict = isinstance(data[0], dict)
    for d in data:
        assert isinstance(d, dict) == isdict
    if isdict:
        return {k: ep.concatenate([d[k] for d in data]) for k in data[0]}
    return ep.concatenate(data)
