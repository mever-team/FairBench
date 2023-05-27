# FairBench
[![codecov](https://codecov.io/gh/mever-team/FairBench/branch/main/graph/badge.svg?token=qeiNv3DN0W)](https://codecov.io/gh/mever-team/FairBench)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Comprehensive AI fairness exploration.

**Author:** Emmanouil (Manios) Krasanakis<br>
**License:**  Apache Software License 2<br>
**Dependencies:** `numpy`,`eagerpy`,`dask.distributed`,`makefun`,`matplotlib`, `pandas`, `scikit-learn`, `wget`<br>

*This project is in its alpha phase of development.*

## Features

:blue_heart: Fairness metrics <br>
:flags: Multimodal & multiattribute <br>
:chart_with_upwards_trend: Reporting<br>
:wrench: ML integration

## Quickstart
First, install the framework with: `pip install --upgrade fairbench`

Let's investigate the fairness of a binary classification algorithm,
like the following. In practice, you will want to separate training 
from test or validation data, as is done in the 
[showcase example](!examples/showcase.ipynb). For a quick start,
let's keep things simple by making predictions over training data:

```python
from sklearn.linear_model import LogisticRegression

x, y = ...

classifier = LogisticRegression()
classifier = classifier.fit(x, y)
yhat = classifier.predict(x)
```

The framework can also be used with other 
machine learning setups: `tensorflow`, `pytorch`, `jax`

Declare a binary sensitive attribute. Either use
a single `sensitive` array,
or consider multiple such attributes, 
which should be set as branches of what
the library calls forks. Forks are
collections of different
variable values for which calculations, such as
fairness assessment, are independently computed
(e.g., in parallel if the distributed mode of
computations is enabled).
Declaring a fork can be done as:

```python
import fairbench as fb

sensitive1, sensitive2 = ...
sensitive = fb.Fork(case1=sensitive1, case2=sensitive2)
```

Non-forked variables (i.e., of normal Python)
are used by all branches of computations, but forked variables
retain different values per branch. You can use any names 
for branches, and you can have as many
as you want. For example, you
can create a fork `fb.Fork(Men=...,Women=...,Other=...)`. 
More details on forks and branches, 
including generation from dicts, intersectionality for
multiattribute fairness, 
and the preferred way of think about multiattribute multivalue forks 
can be found [here](docs/branches.md).

After declaring the protected attribute, generate a
binary fairness assessment for each branch 
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
functionalities, alongside true multiattribute
fairness and customized reports
are described [here](docs/reports.md). 
You can also compute standalone [metrics](docs/metrics.md),
for instance to use them as backpropagateable 
machine learning regularization terms.

:bulb: Prefer calling `fairbench.multireport` for
fairness reporting that does **NOT treat branches independently**.
That reporting hides original branches, performs reductions
to summarize performance and notions of fairness,
and compares branches between themselves. See the report
documentation mentioned above.


## Docs
[Forks and branches](docs/branches.md)<br>
[Reports](docs/reports.md)<br>
[Available metrics](docs/metrics.md)<br>
[Add your own metrics](CONTRIBUTING.md)<br>
[Distributed execution](docs/distributed.md)
