from fairbench.fork import parallel, Fork
import eagerpy as ep


@parallel
def kwargs(**kwargs):
    if not kwargs:
        return None
    return kwargs


@parallel
def concatenate(data1, data2):
    if data1 is None:
        return data2
    if data2 is None:
        return data1
    assert isinstance(data1, dict) == isinstance(data2, dict)
    if isinstance(data1, dict):
        return {k: ep.concatenate([data1[k], data2[k]]) for k in data1}
    return ep.concatenate([data1, data2])
