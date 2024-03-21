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


Here, we show how to investigate the fairness of a binary
classification algorithm. Other types of systems can
be analysed too. Import the library and create some demo
predictions to play with, like below. You can also have
data in popular that ducktype native Python iterables, such as
numpy arrays and pytorch/tensorflow tensors.

```python
import fairbench as fb

test, y, yhat = fb.demos.adult()  # test is a Pandas dataframe
```

Declare sensitive attributes for test data with 
a data structure called [fork](basics/forks.md).
FairBench supports multi-value and multi-attribute 
fairness analysis with the same interfaces by just 
declaring more attributes in the same fork.
Forks can be constructed with many patterns,
depending on available datatypes. A common case is:

```python
sensitive = fb.Fork(fb.categories @ test[8], fb.categories @ test[9])  # analyses of the gender and race columns
sensitive = sensitive.intersectional()  # automatically find non-empty intersections
```

Given that a sensitive attribute fork has been created, 
use it alongside predictions 
to generate a fairness [report](basics/reports.md). 
We now generate a multireport, which compares all population
groups or subgroups pairwise and aggregates all comparisons
to one value for each base performance metric.
We indicate test and prediction data 
with keywords pertaining to the classification task at hand; 
other arguments enable usage of base performance 
[metrics](advanced/metrics.md) for non-classification tasks.

```python
report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report)  # or print(report) or fb.visualize(report) or fb.interactive(report)
```

Perform an [interactive](basics/interactive.md) exploration
of the report to get a sense of where unfairness is exhibited.
This can be done either programmatically or through an interactive UI.

Afterwards, create some stamps about popular fairness definitions 
it adheres to and pack these to a fairness [model card](basics/modelcards.md)
that includes caveats and recommendations.
The snippet below exports the model card to an html file
format and opens it in your browser. It will look like
[this](images/example_modelcard.md).
You can omit the arguments for exporting to a file or 
for immediately showing the modelcards.
You can also export cards in markdown or yaml formats.


```python
stamps = fb.combine(
    fb.stamps.prule(report),
    fb.stamps.accuracy(report),
    fb.stamps.four_fifths_rule(report)
)
fb.modelcards.tohtml(stamps, file="output.html", show=True)
```


!!! danger
    Always consult stakeholders to decide which stamps are
    relevant for your systems. If a report reveals potential
    issues that do not match those of model cards, also
    ask about them (and you can create custom stumps).
