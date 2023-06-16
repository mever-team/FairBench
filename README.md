# FairBench

[![codecov](https://codecov.io/gh/mever-team/FairBench/branch/main/graph/badge.svg?token=qeiNv3DN0W)](https://codecov.io/gh/mever-team/FairBench)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive AI fairness exploration framework.

**Author:** Emmanouil (Manios) Krasanakis <br>
**License:** Apache Software License 2

## Features

- :blue_heart: Fairness metrics
- :flags: Multivalue multiattribute
- :chart_with_upwards_trend: Reporting
- :wrench: ML integration (`numpy`,`torch`,`tensorflow`,`jax`)

## Quickstart

Install the framework with:

```shell
pip install --upgrade fairbench
```

To investigate the fairness of a binary classification algorithm, follow these steps:

1. Import the library and load your data:

```python
import fairbench as fb

trainx, trainy, x, y = ...
```

2. Create some predictions, for example after training a model:

```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(trainx, trainy)
yhat = classifier.predict(x)
```

3. Declare sensitive attributes from binary columns:

```python
sensitive1, sensitive2 = ...
sensitive = fb.Fork(case1=sensitive1, case2=sensitive2)
```

4. Generate a report (more advanced reports have the same interface) and show it:

```python
report = fb.binreport(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report)  # or print(report) or fb.visualize(report)
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
This reporting hides original branches, compares 
branches with each other, and performs reductions. 
Refer to the documentation below for more details.


## Docs
[<img alt="reports" width="30%" src="docs/reports.png" />](docs/reports.md) 
[<img alt="branches" width="30%" src="docs/forks.png" />](docs/forks.md) 

Advanced topics:
- [Available metrics](docs/metrics.md)
- [Add your own metrics](CONTRIBUTING.md)
- [Distributed execution](docs/distributed.md)

Example usage:
- [Graphs](examples/graphs.ipynb) <small>- node recommendation, algorithm comparison</small>
- [Tabular](examples/demo.ipynb) <small>- classification, intersectionality</small>
- [Vision](examples/vision.ipynb) <small>- multiclass</small>
