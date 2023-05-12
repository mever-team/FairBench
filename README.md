# FairBench
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Bringing together existing and new frameworks for fairness exploration.

**This project is in its alpha phase.**

**Dependencies:** `numpy`,`eagerpy`,`dask.distributed`,`makefun`,`matplotlib`


## Features

:blue_heart: Fairness-aware metrics <br>
:checkered_flag: Multi-modal and multi-objective <br>
:chart_with_upwards_trend: Reporting<br>
:wrench: Backpropagateable <br>
:satellite: Parallel/distributed

## Quickstart
First, install the framework with: `pip install --upgrade fairbench`

Let's investigate the fairness of a binary classification algorithm,
like the following:

```python
from sklearn.linear_model import LogisticRegression

x, y = ...

classifier = LogisticRegression()
classifier = classifier.fit(x, y)
yhat = classifier.predict(x)
```

The framework can also be used with other 
machine learning setups: `tensorflow`, `pytorch`, `jax`

Declare a binary sensitive attribute; either use
a single `sensitive` array or consider multiple such attributes,
which should be set as a data fork per:

```python
import fairbench as fb

sensitive1, sensitive2 = ...
sensitive = fb.Fork(case1=sensitive1, case2=sensitive2)
```

Variable forks create branches of calculations that are computed
in parallel. Non-forked variables (i.e., of normal Python)
are used by all branches of computations, but forked variables
retain different values per branch. More details on forks and branches, 
including generation from dicts, can be found [here](docs/branches.md).

After declaring the protected attribute, generate a
binary assessment on each branch's fairness 
and print it per:

```python
report = fb.binreport(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report)
```

A different value is printed per branch:

```
                case1           case2          
accuracy        0.938           0.938          
prule           0.571           0.571          
dfpr            -0.071          0.071          
dfnr            -0.500          0.500  
```

Omitting some arguments from the report will 
prevent some measures that depend on them 
from being evaluated. The generated report can also 
be visualized, exported as *json*,
or be constrained to specific metrics. These 
functionalities, alongside true multi-attribute
fairness and customized reports
are described [here](docs/reports.md). 
You can also compute standalone [metrics](docs/metrics.md),
for instance to use them as backpropagateable 
machine learning regularization terms.

:warning: Prefer calling `fairbench.multireport` for
fairness reporting that does **NOT treat branches independently**.
That reporting hides original branches, performs reductions
to summarize performance and notions of fairness,
and compares branches between themselves. See the report
documentation mentioned above.


## Docs
[Variable branches](docs/branches.md)<br>
[Reports](docs/reports.md)<br>
[Available metrics](docs/metrics.md)<br>
[Add your own metrics](CONTRIBUTING.md)<br>
[Distributed execution](docs/distributed.md)
