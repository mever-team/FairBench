# FairBench

[![codecov](https://codecov.io/gh/mever-team/FairBench/branch/main/graph/badge.svg?token=qeiNv3DN0W)](https://codecov.io/gh/mever-team/FairBench)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive AI fairness exploration framework.

**Author:** Emmanouil (Manios) Krasanakis
**License:** Apache Software License 2

## Features

- :blue_heart: Fairness metrics
- :flags: Multivalue multiattribute
- :chart_with_upwards_trend: Reporting
- :wrench: ML integration

## Quickstart

Install the framework with:

```shell
pip install --upgrade fairbench
```

To investigate the fairness of a binary classification algorithm, follow these steps:

1. Import the library and load your data:

```python
import fairbench as fb
from sklearn.linear_model import LogisticRegression

x, y = ...
```

2. Train a binary classification model:

```python
classifier = LogisticRegression()
classifier.fit(x, y)
yhat = classifier.predict(x)
```

3. Declare sensitive attributes from binary columns:

```python
sensitive1, sensitive2 = ...
sensitive = fb.Fork(case1=sensitive1, case2=sensitive2)
```

4. Generate a binary fairness assessment and print it:

```python
report = fb.binreport(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report)
```

The output will display the assessment for each branch:

```
            case1           case2
accuracy    0.938           0.938
prule       0.571           0.571
dfpr        -0.071          0.071
dfnr        -0.500          0.500
```

:bulb: For fairness reporting that does 
**NOT treat branches independently**, 
it is recommended to use `fairbench.multireport`. 
This reporting hides original branches, performs reductions 
to summarize performance and notions of fairness, and compares 
branches with each other. Refer to the report documentation 
mentioned above for more details.


## Docs
More information:

- [Reports](docs/reports.md) <small>- combine, explain, export, visualize</small>
- [Forks and branches](docs/branches.md) <small>- multiattribute multivalue sensitive attributes</small>

Advanced topics:
- [Available metrics](docs/metrics.md)
- [Add your own metrics](CONTRIBUTING.md)
- [Distributed execution](docs/distributed.md)

