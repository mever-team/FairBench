# Parallel/distributed computing

`FairBench` supports parallel or distributed computing
of computationally intensive operations that are separated
into different [branches](branches.md)
via [dask](https://www.dask.org).
This capability can be enabled per:

```python
import fairbench as fb

fb.distributed(*args, **todict)
```

where the arguments and keyword arguments are those
necessary to instantiate a `dask.Client`. For example,
you can provide no arguments to start simple parallel
computing, where workers are created locally in your machine.
You can also provide an IP address pointing to the dask
server. If the server's workers have been instantiated with the same
names as some branches, those branches will be executed
there.

:warning: If computations are too simple, parallelization
will be slower that non-parallelization,
because a significant overhead is needed to transfer data to workers.
