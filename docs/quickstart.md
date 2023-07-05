# Quickstart

Install FairBench with:

```shell
pip install --upgrade fairbench
```

We will show how to investigate the fairness of a binary
classification algorithm - other types of predictions
are computable too. To work with some data, import
the library and create some predictions as shown below:

```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(trainx, trainy)
yhat = classifier.predict(x)
```

Declare sensitive attributes for test data. This is
done with a data structure called [fork](basics/forks.md), 
and which model additional multivalue and multiattribute 
considerations with the same interface.
Fork construction can be simplied, depending on your datatypes.
Then, use the predictions and the fork
to generate a fairness[report](basics/reports.md). 
Here, we will generate a multireport, which is the most general 
deterministic fairness exploration FairBench provides.

```python
sensitive1, sensitive2 = ...
sensitive = fb.Fork(case1=sensitive1, case2=sensitive2)
report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report)  # or print(report) or fb.visualize(report) or fb.interactive(report)
```

!!! info
    If you only have one binary sensitive attribute
    or want to consider some older fairness analysis
    measures, create a binreport instead.

Either perform an [interactive](basics/interactive.md) exploration
of the report to get a sense of where unfairness is exhibited, or
use create some stamps about fairness definitions it adheres to
and use these to generate a fairness modelcard:

```python
stamps = fb.combine(
    fb.stamps.prule(report),
    fb.stamps.accuracy(report),
    fb.stamps.four_fifths_rule(report)
)
fb.modelcards.tohtml(stamps, show=True)
```