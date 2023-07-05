# Reports

1. [Generating reports](#generating-reports)
2. [Report types](#report-types)
3. [Viewing reports](#viewing-reports)
4. [Editing reports](#editing-reports)
5. [Reducing reports](#reducing-reports)
6. [Combining reports](#combining-reports)
7. [Explaining report values](#explaining-report-values)
8. [Buffering batch predictions](#buffering-batch-predictions)

## Generating reports

You can generate 
fairness-aware reports by providing some
(preferably all) of the following arguments
of the `fairbench.report(...)` function:

* binary `predictions`
* ideal binary prediction `labels`
* binary `sensitive` attribute

In addition to the above, you need to provide
a mandatory `metrics` argument that holds either 
a dictionary mapping metric names to metric functions
or a list of metric functions, where
in the last case their names are inferred
from the function names.
You can use variable [forks](forks.md)
to handle multi-value attributes or multiple
sensitive attribute values. 
For the time being, you need
to create a different branch for each
sensitive and predictive attribute
combination, but this will likely be
automated in the future.

## Report types

Out-of-the box, you can use one of the following three
report generation methods, which wrap the base
report generation:
- `fairbench.accreport` provides popular performance evaluation measures to be viewed between branches (accuracy, positives, true positive rates, true negative rates).
- `fairbench.binreport` conducts a suit of popular binary fairness assessments on each variable branch and should be preferred when branches do *not* correspond to mult-attribute fairness.
- `fairbench.multireport` is ideal for multi-fairness approaches, where the `sensitive` argument is a fork. This report generator performs a lot of [editing](../advanced/manipulation.md) to summarize the findings of multi-attribute fairness analysis.
- `fairbench.isecreport` tackles mult-fairness with many intersectional groups. Its output approximates *multireport* with a Bayesian framework that is applicable even when protected group intersections are too small to yield meaningful predictions.

As an example, let's create a simple report
based on binary predictions, binary
ideal predictions and multiclass
sensitive attribute `sensitive`, which is
declared to be a fork with two branches
*case1,case2*, each of which is a binary
feature value per:

```python
import fairbench as fb

sensitive = fb.Fork(case1=..., case2=...)
report = fb.multireport(predictions=..., labels=..., sensitive=sensitive)
```


## Viewing reports

The report is actually a fork for each data branch
(for case1 and case2) that holds a dictionary of
metric results. Several methods are provided to
work with this data format. First, you can print 
the report on the *stdout* console:

```python
fb.describe(report)  
```

```
Metric          mean            minratio        maxdiff        
accuracy        0.938           1.000           0.000          
pr              0.812           0.857           0.125          
fpr             0.063           0.778           0.016          
fnr             0.333           0.333           0.333  
```

But you can also convert the report to a *json*
format, for example to send to your frontend:

```python
print(fb.tojson(report))
```

```json
{"header": ["Metric", "mean", "minratio", "maxdiff"], "accuracy": [0.9375, 1.0, 0.0], "pr": [0.8125, 0.8571428571428571, 0.125], "fpr": [0.06349206349206349, 0.7777777777777778, 0.015873015873015872], "fnr": [0.3333333333333333, 0.3333333333333333, 0.33333333333333337]}
```

Reports can be visualized  with `matplotlib`:
```python
fb.visualize(report)
```


![report example](reports.png)

!!! warning 
    Reports are limited to forks of metrics. You can print any
    fork, but  `fb.visualize` and `fb.display`  only work 
    reports. For example, you cannot visualize a fork 
    of reports, though you can obtain a view of it and visualize that. 
    To explore complicated forks, use [interactive visualization](interactive.md).


## Explaining report values

Some report values, especially those procured by 
[reduction](../advanced/manipulation.md) in multireports,
can be explained in terms of data they are derived from.
For instance, if a `fairbench.isecreport` is made, both
empirical and bayesian evaluations arise from the underlying
data branches of multi-attribute fairness forks.

Whenever possible, the data branches that are converted
into final reports are preserved by having report values
be instances of the `Explainable` class.
This provides an `.explain` field of data contributing
to the report value, and `.desc` field to store additional 
descriptions.

As an example, here we use these fields
to retrieve posterior estimations contributing to
calculating the *baysian* branch of the minprule
metric in the *isecreport*:

```python
report = fb.isecreport(vals)
fb.describe(report)
fb.describe(report.bayesian.minprule.explain)
```
```
Metric          empirical       bayesian       
minprule        0.857           0.853          

Metric          case1           case2           case2,case1    
                0.729           0.706           0.827     
```

Similarly, you can obtain explanations about the values
contributing to report combinations, for instance
when comparing two algorithms.


## Buffering batch predictions

When training machine learning algorithms, you may want
to concatenate the same variable generated across 
several batches. This can be used by calling
`fairbench.todict` to convert a set of keyword arguments
to a fork of dictionaries. Entries of such forks (e.g.,
`predictions` in the example below) can be concatenated
via a namesake method. Concatenate such dictionaries
with previous concatenation outcomes to generate a
dictionary of keyword arguments to pass to reports like so:

```python
data = None
for batch in range(batches):
    yhat, y, sensitive = ...  # compute for the batch
    data = fb.concatenate(data, fb.todict(predictions=yhat, labels=y, sensitive=sensitive))
report = fb.binreport(data)
```