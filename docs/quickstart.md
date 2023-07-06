# Quickstart

Install FairBench with:

```shell
pip install --upgrade fairbench
```

We will show how to investigate the fairness of a binary
classification algorithm - other types of systems can
be analysed too. To work with some data, import
the library and create some predictions:

```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(trainx, trainy)
yhat = classifier.predict(x)
```

Declare sensitive attributes for test data with 
a data structure called [fork](basics/forks.md).
This also lets the FairBench assess multi-value and multi-attribute 
fairness with the same interfaces.
Fork construction can be simplified, depending on available datatypes.

Given that a sensitive attribute fork has been created, 
use the predictions and the fork
to generate a fairness [report](basics/reports.md). 
Here, we will generate a multireport, which is the most general 
-albeit a little complex- 
deterministic fairness exploration FairBench provides. We provide
some test and predictive data arguments with keywords pertaining
to the classification task at hand; other arguments could lead to 
the computation of fairness based on different metrics.

```python
sensitive = fb.Fork(men=sensitive1, case2=sensitive2)
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
and use these to generate a fairness modelcard. 
This snippet below exports a model card based to a file in html
format and shows it in your browser.
You could omit the file export or showing arguments, and could
even export cards in markdown or yaml formats.


```python
stamps = fb.combine(
    fb.stamps.prule(report),
    fb.stamps.accuracy(report),
    fb.stamps.four_fifths_rule(report)
)
fb.modelcards.tohtml(stamps, file="output.html", show=True)
```
