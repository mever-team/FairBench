# :chart_with_upwards_trend: Reports

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
to handle multi-attribute values. 
For the time being, you need
to create a different branch for each
sensitive and predictive attribute
combination, but this will likely be
automated in the future.


Out-of-the box, you can use one of the following three
report generation methods:
- `fairbench.accreport` provides popular performance evaluation measures to be viewed between branches.
- `fairbench.binreport` conducts a suit of popular binary fairness assessments on each variable branch and should be preferred when branches do *not* correspond to mult-attribute fairness.
- `fairbench.multireport` is ideal for mult-fairness approaches, where the `sensitive` argument is a fork. This report generator performs a lot of the [report editing](#editing-reports) described below to summarize the findings of multi-attribute fairness.

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

This will output the following to the console:
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

This will print the following:
```
{"header": ["Metric", "mean", "minratio", "maxdiff"], "accuracy": [0.9375, 1.0, 0.0], "pr": [0.8125, 0.8571428571428571, 0.125], "fpr": [0.06349206349206349, 0.7777777777777778, 0.015873015873015872], "fnr": [0.3333333333333333, 0.3333333333333333, 0.33333333333333337]}
```

Finally, reports can be visuzualized by calling:
```python
fb.visualize(report)
```

This will generate the following figure:

![report example](reports.png)

## Editing reports
Since reports are forks of dictionaries, you can use normal
dictionary methods to access and edit their elements (given
that forks provide access to any possible methods of internal
objects). For instance, you can use the following code
to calculate a notion of total disparate mistreatment as the sum
of *dfnr* and *dfpr* of a binary report in all brnaches
and remove these entries from all branch
dictionaries using Python's `dict.pop` method:

```python
import fairbench as fb

sensitive = fb.Fork(case1=..., case2=...)
report = fb.binreport(predictions=..., labels=..., sensitive=sensitive)
fb.describe(report)

report["mistreatment"] = abs(report["dfpr"]) + abs(report["dfnr"])
report.pop("dfpr")
report.pop("dfnr")
fb.describe(report)
```

This will print the following to teh console:
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

In addition to transforming results in this manner, reports
can also be reduced or combined alongside branches. Again,
these operations are applicable to all variable forks.
However, this time their usage is discouraged outside of 
report manipulation, as they create new -and therefore potentially 
unforeseen- data branches. Still, these constitute the main mechanism
for handling multi-attribute reports.

Reduction internally runs three types of functions obtained
from its arguments:
- `transform` values found in the report for each metric, which can be either *None* or *abs*.
- `expand` the list of branch values for each metric, namely *None*, a pairwise *ratio* between values, or absolute *diff*erences between branch values.
- `reducer` method that takes a list of all branch values for each metric and summarizes them into one value. These can be *mean,max,min,sum,budget*, where the last one is the logarithm of the maximum declared in differential fairness formulations.

To demonstrate reduction,
we compute the mean, and budget of the absolute value ratio
via the following code:

```python
import fairbench as fb

sensitive = fb.Fork(case1=..., case2=...)
report = fb.accreport(predictions=..., labels=..., sensitive=sensitive)

mean_across_branches = fb.reduce(report, fb.mean, name="avg")
max_abs_across_branches = fb.reduce(report, fb.budget, expand=fb.ratio, transform=fb.abs)
```
 
Recuctions create new reports that comprise only one branch.
The branch's name is dynamically derived by parameters 
(e.g., *"budgetratioabs"*), but you can also use the `name` 
argument to set a specific name instead.

Reports, including reduced ones, can be combined alongside 
branches, as demonstrated in the following snippet:

```python
new_report = fb.combine(report, mean_across_branches, max_abs_across_branches)
fb.describe(new_report)
```

This will output the following to the console:
```
Metric          case2           case1           avg             budgetratioabs 
accuracy        0.938           0.938           0.938           0.000          
fpr             0.056           0.071           0.063           0.251          
fnr             0.167           0.500           0.333           1.099    
```
