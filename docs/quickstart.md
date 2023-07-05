# Quickstart

Install FairBench with:

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
fb.describe(report)  # or print(report) or fb.visualize(report) or fb.interactive(report)
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
