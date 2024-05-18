# Quickstart

Before starting, install FairBench with:

```shell
pip install --upgrade fairbench
```

A typical workflow using the library will help you identify prospective
fairness concerns to discuss with stakeholders. You need to eventually
decide which of the concerns matter after getting a wide enough
picture. Follow the steps below.

## 1. Prepare test data

Run your system to generate some predictions for test data.
Here, we assess biases for a binary
classifier, but other types of predictions can be analysed too. 
Supported data formats
include lists, numpy arrays, and pytorch/tensorflow tensors.
Import the library and create some demo
predictions to play with:

```python
import fairbench as fb
test, y, yhat = fb.demos.adult()  # test is a Pandas dataframe
```

## 2. Set    sensitive attribute [fork](basics/forks.md)

Pack sensitive attributes found in your test data
into a data structure called fork.
This can store any number of attributes with any number of values
by considering each value as a separate dimension.
A fork can be constructed with many patterns, like this one:

```python
sensitive = fb.Fork(fb.categories @ test[8], fb.categories @ test[9])  # analyses of the gender and race columns
sensitive = sensitive.intersectional()  # automatically find non-empty intersections
```

## 3. Explore fairness [reports](basics/reports.md)

Use sensitive attribute forks alongside predictions 
to generate fairness reports.
Next we generate a multireport, which compares all population
groups or subgroups pairwise based on some base performance metrics
and aggregates all comparisons to one value.
The task type (here: binary classification)
and corresponding base performance 
metrics are determined by
which arguments are provided.

```python
report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report)  # or print(report) or fb.visualize(report) or fb.interactive(report)
```

[Explore](basics/interactive.md) 
reports by backtracking their
intermediate computations
to get a sense of where unfairness comes from.
This can be done either programmatically 
or through an interactive UI
that is also launched programmatically via `fb.interactive(report)`.

## 4. Create fairness [model cards](advanced/modelcards.md)

After determining key issues at play,
create some stamps of popular fairness definitions 
and organize these into a fairness model card.
that includes caveats and recommendations.
The snippet below creates a card like
[this one](images/example_modelcard.md).
You can omit the arguments for exporting to a file or 
for immediately showing the modelcards.
You can also export to markdown or yaml formats.


```python
stamps = fb.combine(
    fb.stamps.prule(report),
    fb.stamps.accuracy(report),
    fb.stamps.four_fifths(report)
)
fb.modelcards.tohtml(stamps, file="output.html", show=True)
```


!!! danger
    Blindly stamping systems is not 
    always a good idea. Consult with stakeholders to determine
    what actually matters.
