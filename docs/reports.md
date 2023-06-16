# :chart_with_upwards_trend: Reports

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
You can use variable [forks](branches.md)
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
- `fairbench.accreport` provides popular performance evaluation measures to be viewed between branches (accuracy, positives, true positive rates, true negative rates). These values are useful to use as input to reductions when [generating reports](#generating-reports).
- `fairbench.binreport` conducts a suit of popular binary fairness assessments on each variable branch and should be preferred when branches do *not* correspond to mult-attribute fairness.
- `fairbench.multireport` is ideal for multi-fairness approaches, where the `sensitive` argument is a fork. This report generator performs a lot of the [report editing](#editing-reports) described below to summarize the findings of multi-attribute fairness.
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

```
{"header": ["Metric", "mean", "minratio", "maxdiff"], "accuracy": [0.9375, 1.0, 0.0], "pr": [0.8125, 0.8571428571428571, 0.125], "fpr": [0.06349206349206349, 0.7777777777777778, 0.015873015873015872], "fnr": [0.3333333333333333, 0.3333333333333333, 0.33333333333333337]}
```

Finally, reports can be visualized by calling:
```python
fb.visualize(report)
```

![report example](reports.png)

## Editing reports
Since reports are forks of dictionaries, you can use normal
dictionary methods to access and edit their elements (given
that forks provide access to any possible methods of internal
objects). For instance, you can use the following code
to calculate a notion of total disparate mistreatment as the sum
of *dfnr* and *dfpr* of a binary report in all brnaches
and remove these entries from all branch
dictionaries using Python's dictionary entry deletion:

```python
import fairbench as fb

sensitive = fb.Fork(case1=..., case2=...)
report = fb.binreport(predictions=..., labels=..., sensitive=sensitive)
fb.describe(report)

report["mistreatment"] = abs(report.dfpr) + abs(report.dfnr)
del report["dfpr"]
del report["dfnr"]
fb.describe(report)
```

This will print the following to the console:
```
Metric          case2           case1          
accuracy        0.938           0.938          
prule           0.667           0.571          
dfpr            0.056           0.071          
dfnr            0.167           0.500          

Metric          case2           case1          
accuracy        0.938           0.938          
prule           0.667           0.571          
mistreatment    0.222           0.571
```


## Reducing reports

Reports can also be reduced alongside branches. Again,
this operation is applicable to all variable forks,
although this time usage is discouraged outside of 
report manipulation, as reduction creates new -and therefore potentially 
unforeseen- data branches, but constitutes the main mechanism
for summarizing multi-attribute reports into one measure.
Reduction internally runs three types of functions obtained
from its arguments:
- `transform` values found in the report for each metric, which can be either *None* or *abs*.
- `expand` the list of branch values for each metric, namely *None*, a pairwise *ratio* between values, or absolute *diff*erences between branch values.
- `reducer` method that takes a list of all branch values for each metric and summarizes them into one value. These can be *mean,max,min,sum,budget*, where the last one is the logarithm of the maximum declared in differential fairness formulations.

To demonstrate usage,
we compute the mean, and budget of the absolute value ratio
via the following code:

```python
import fairbench as fb

sensitive = fb.Fork(case1=..., case2=...)
report = fb.accreport(predictions=..., labels=..., sensitive=sensitive)

mean_across_branches = fb.reduce(report, fb.mean, name="avg")
max_abs_across_branches = fb.reduce(report, fb.budget, expand=fb.ratio, transform=fb.abs)
```

:warning: You will typically want to perform custom reductions on
an *accreport* or on manually generated reports for some non-fairness-related
evaluation measures. You can then combine the outcome of some reductions
to provide a holistic view.
 
Recuction creates new reports that comprise only one branch.
The branch's name is dynamically derived by parameters 
(e.g., *"budgetratioabs"*), but you can also use the `name` 
argument to set a specific name instead. Set `name=None` 
to directly retrieve report outputs instead of putting them
to a Fork.

# Combining reports
Reports, including reduced ones, can be combined to
create a super-report with all sub-branches. This 
is demonstrated in the following snippet:

```python
new_report = fb.combine(report, mean_across_branches, max_abs_across_branches)
fb.describe(new_report)
```

```
Metric          case2           case1           avg             budgetratioabs 
accuracy        0.938           0.938           0.938           0.000          
fpr             0.056           0.071           0.063           0.251          
fnr             0.167           0.500           0.333           1.099    
```

Sometimes, you may want to compare the same report
generated for multiple algorithms. To do this, you
need to generate a fork where each branch holds
a respective algorithm's default. Then, you can 
extract and combine values for each metric as
shown in the following snipper:

```python
reports = fb.Fork(ppr=report, lfpro=fair_report)
rep = fb.extract(acc=reports.mean.accuracy, prule=reports.minratio.pr)
fb.describe(rep)
```

```
Metric          ppr             lfpro          
acc             0.819           0.826          
prule           0.261           0.957    
```

When extracting values from reports, you can optionally
omit the final value getter as long as is the same as the
new name. For example, `fb.extract(accuracy=reports.mean)`
is equivalent to `fb.extract(accuracy=reports.mean.accuracy)`
given that `reports.mean` also returns a fork.


## Explaining report values

Some report values, especially those procured by reduction,
can be explained in terms of data they are derived from.
For instance, if a `fairbench.isecreport` is made, both
empirical and bayesian evaluations arise from the underlying
data branches of multi-attribute fairness forks.
Whenever possible, the data branches that are converted
into final reports are preserved by having report values
be instances of an `fairbench.Explainable` class.
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