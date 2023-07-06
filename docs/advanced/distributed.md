# Computational branches

1. [Multiple forked variables](#multiple-forked-variables)
2. [Parallel & distributed computing](#parallel--distributed-computing)

## Multiple forked variables

If you have multiple [forks](../basics/forks.md),
they should all have the same branches.
Each branch will execute independently 
of the rest (some variable may be shared).
You can even create machine learning model 
forks, where a different model is applied 
on different branches:

```python
from sklearn.linear_model import LogisticRegression, MLPClassifier

x, y = ...

classifier = fb.Fork(case1=LogisticRegression(), case2=MLPClassifier())
classifier = classifier.fit(x, y)
yhat = classifier.predict(x)
```

Forks automatically try to call wrapped class methods,
i.e., `classifier.fit` is also a fork whose branches
hold the outcome of applying `fit` on each branch's model.
The inputs `x,y` could also have been forks, 
in which case each branch would have been trained on
respective values.

Recall that branch values can be accessed via class fields,
for example like `yhat = (yhat.case1+yhat.case2)/2`. This 
computation produces a factual value that is not
bound to any branch. On the other hand `yhat.case1`
and `yhat.case2` would be used during assessment of
case1 sensitive attribute values and case2 sensitive
attribute values. 

```python
print(classifier)
# case2: MLPClassifier()
# case1: LogisticRegression()
print(yhat)
# case2: [0 1 0 1 0 1 1 1]
# case1: [0 1 0 1 1 1 1 1]
print((yhat.case1+yhat.case2)/2)
# [0.  1.  0.  1.  0.5 1.  1.  1. ]
```

!!! danger 
    Avoid overlapping names between branches 
    and class fields or methods, as they are both 
    accessed with the same annotation.
    If there is confusion, branch values will be obtained.

A visual view of how data 
are organized across branches follows. Some
variables are identical but others
obtain different values per branch. The same
code is run on all branches concurrently and
independently.

![branches](branches.png)

!!! tip
    Use branches to run several computation pipelines concurrently.

## Parallel & distributed computing

FairBench supports parallel or distributed computing
of computationally intensive operations that are separated
into different branches
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
there in respective servers.

!!! warning 
    If computations are too simple, parallelization
    will be slower, due to data transfer overheads.

!!! warning
    Accessing branch values, for instance in report generation 
    and visualization, under distributed computing
    awaits for dependent remote computations to conclude.
