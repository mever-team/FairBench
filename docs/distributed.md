# :satellite: Parallel/distributed computing

`FairBench` supports parallel or distributed computing
of computationally intensive operations that are separated
into different [branches](branches.md)
via [dask](https://www.dask.org).


This capability can be enabled per:

```python
import fairbench as fb

fb.distributed(*args, **kwargs)
```

where the arguments and keyword arguments are those
necessary to instantiate a `dask.Client`. For example,
you can provide no arguments to start simple parallel
computing, where workers are created locally in your machine,
or you can provide an IP address pointing to the dask
server.

If server workers have been instantiated with the same
names as some branches, those branches will be executed
there.

:warning: If your computations are too simple, parallelization
will be slower that non-parallelization,
as a significant overhead is needed to transfer data to workers.