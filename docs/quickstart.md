# Quickstart

Install FairBench with:

```shell
pip install --upgrade fairbench
```

A typical workflow involves the following steps:
1. Produce test or validation data
2. Declare (multi-attribute multi-value) sensitive attributes as [forks](basics/forks.md)
3. Create and explore [reports](basics/reports.md) that present many types of biases
4. Extract relevant fairness definitions into [model cards](basics/modelcards.md)


We will show how to investigate the fairness of a binary
classification algorithm. Other types of systems can
be analysed too. Import
the library and create some predictions:

```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(trainx, trainy)
yhat = classifier.predict(x)
```

Declare sensitive attributes for test data with 
a data structure called [fork](basics/forks.md).
FairBench supports multi-value and multi-attribute 
fairness analysis with the same interfaces by just adding more
attributes to the same fork.
Forks can be constructed with many patterns, depending on available datatypes.

Given that a sensitive attribute fork has been created, 
use it alongside predictions 
to generate a fairness [report](basics/reports.md). 
Here, we will generate a multireport, which is the most general 
-albeit a little complex- 
deterministic fairness exploration FairBench provides. We set
some test and predictive data arguments with keywords pertaining
to the classification task at hand; other arguments enable 
[metrics](advanced/metrics.md) for other tasks.

```python
sensitive = fb.Fork(men=sensitive1, case2=sensitive2)
report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report)  # or print(report) or fb.visualize(report) or fb.interactive(report)
```

Either perform an [interactive](basics/interactive.md) exploration
of the report to get a sense of where unfairness is exhibited, or
create some stamps about fairness definitions it adheres to
and use these to generate a fairness [model card](basics/modelcards.md). 
The snippet below exports a model card to an html file
format and opens it in your browser. The card will look like
[this](images/example_modelcard.md).
You can omit the file export or show arguments, or
export cards in markdown or yaml formats.


```python
stamps = fb.combine(
    fb.stamps.prule(report),
    fb.stamps.accuracy(report),
    fb.stamps.four_fifths_rule(report)
)
fb.modelcards.tohtml(stamps, file="output.html", show=True)
```


!!! tip
    Multireport considers many possible variations
    of what could constitute bias. For naive analysis, 
    create a binreport instead.

!!! danger
    Always consult stakeholders to decide which stamps are
    relevant for your systems.
