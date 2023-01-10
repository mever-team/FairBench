# FairBench
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Bringing together existing and new frameworks for fairness exploration.

**This project is in its pre-alpha phase.**

**Dependencies:** `numpy`, `eagerpy`, `dask.distributed`, `makefun`, `matplotlib`


## Features

:blue_heart: Fairness-aware metrics <br>
:checkered_flag: Multi-modal and multi-objective <br>
:chart_with_upwards_trend: Reporting<br>
:wrench: Backpropagatable <br>
:satellite: Parallel/distributed

## :rocket: Quickstart
First, install the framework with: `pip install --upgrade fairbench`

Create some binary classification algorithm like the following:

```python
import fairbench as fb
from sklearn.linear_model import LogisticRegression

x, y = ...

classifier = LogisticRegression()
classifier = classifier.fit(x, y)
yhat = classifier.predict(x)
```

The fairness assessment framework can also be used with other 
machine learning setups: `tensorflow`, `pytorch`, `jax`

We now declare a binary sensitive attribute. We can either use
a single `sensitive` array or consider multiple such attributes,
which we se as a variable fork per:

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


## Docs
[Data branches](docs/branches.md)<br>
[Add your own metrics](CONTRIBUTING.md)
