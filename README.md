# FairBench
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Bringing together existing and new frameworks for fairness exploration.

**This project is in its pre-alpha phase.**

**Dependencies:** `numpy`,`eagerpy`, `dask.distributed` (optional)

## :rocket: Quickstart

First, let's train a binary classification algorithm:

```python
import fairbench as fb
from sklearn.linear_model import LogisticRegression

x, y = ...

classifier = LogisticRegression()
classifier = classifier.fit(x, y)
yhat = classifier.predict(x)
```

Let us now consider some sensitive attribute. We can either use
a single `sensitive` array or consider multiple such attributes,
which we declare as a variable fork per:

```python
sensitive1, sensitive2 = ...
sensitive = fb.Fork(case1=sensitive1, case2=sensitive2)
```

Variable forks create branches of calculations that are computed
in parallel. Non-forked variables (i.e., of normal Python)
are used by all branches of computations, but forked variables
retain different values per branch.

After declaring the protected attribute, we generate a
report on the branch's fairness and print it per:

```python
report = fb.report(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report)
```

A different value is printed per branch:

```
                case1           case2          
accuracy        0.938           0.938           
dfnr            0.500           -0.167          
dfpr            0.071           -0.100          
prule           0.571           0.833    
```

:warning: Omitting some arguments from the report will 
prevent some measure from being evaluated.

## :brain: Advanced usage
Depending on the modality being assessed, 
you can have different training or test data, 
or you might want to use different predictive models.
For instance, in the above example you can write:

```python
classifier = fb.Fork(case1=LogisticRegression(), case2=LogisticRegression(tol=1.E-6))
```
which makes a different classifier's outcome be assessed 
in each case.

Accessing modal values -if present- is done as class fields,
for example like `yhat = (yhat.case1+yhat.case2)/2`. This 
computation produces a factual tensor that is not
bound to any modality, but if was not there `yhat.case1`
and `yhat.case2` would be used during assessment of
case1 sensitive attribute values and case2 sensitive
attribute values. Here is a visual view of how data 
are organized between branches:

![branches](branches.png)

:bulb: You can use branches to run several computations
pipelines concurrently.

## :pencil: Contributing
`FairBench` was designed to be easily extensible.

<details>
<summary>Create a new metric.</summary>

1. Fork the repository.
2. Create it under `fairbench.metrics` module.
3. Add the `parallel` decorator like this:
```
from faibench import parallel

@parallel
def metric(...):
    return ...
```
3. Reuse as many arguments found in other metrics as possible. 
4. Write tests and push the changes in your fork.
5. Create a pull request from github's interface.

:warning: Numeric inputs are automatically converted into 
[eagerpy](https://github.com/jonasrauber/eagerpy) tensors,
which use a functional interface to ensure interoperability.
You can use the `@parallel_primitive` decorator to avoid
this conversion and work with specific primitives users of the
framework provide, but try not to do so. 

:bulb: If your metric should behave differently for different 
data branches, add a `branch=None` default argument in its
definition to get that branch.

</details>
