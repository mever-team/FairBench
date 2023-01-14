# :chart_with_upwards_trend: Reports

## Generating reports

In the simplest case, you can generate 
fairness-aware reports by providing some
(preferably all) of the following arguments
of the `fairbench.report(...)` function:
* binary `predictions`
* ideal binary prediction `labels`
* binary `sensitive` attribute

You can use variable [forks](branches.md)
to handle multi-attribute values. 
For the time being, you need
to create a different branch for each
sensitive and predictive attribute
combination, but this will likely be
automated in the future.
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
report = fb.report(predictions=..., labels=..., sensitive=sensitive)
```

## Viewing reports

The report is actually a fork for each data branch
(for case1 and case2) that holds a dictionary of
metric results. Several methods are provided to
work with this data format. First, you can print 
the report on the *stdout* console:

```python
fb.describe(report)
#                 case1           case2          
# accuracy        0.938           0.938           
# dfnr            0.500           -0.167          
# dfpr            0.071           -0.100          
# prule           0.571           0.833    
```

But you can also convert the report to a *json*
format, for example to send to your frontend:

```python
print(fb.json(report))
#  {"header": ["Metric", "case2", "case1"], 
#   "accuracy": [0.9375, 0.9375], 
#   "dfnr": [0.16666666666666666, 0.5], 
#   "dfpr": [0.05555555555555555, 0.07142857142857142], 
#   "eqrep": [0.2222222222222222, 0.5714285714285714], 
#   "prule": [0.6666666666666666, 0.5714285714285714]}
```

Finally, reports can be graphically presented
by calling:
```python
fb.visualize(report)
```

## Editing reports
Since reports are forks of dictionaries, you can use normal
dictionary methods to access and edit their elements (given
that forks provide access to any possible methods of internal
objects). For instance, you can use the following code
to calculate a notion of total disparate mistreatment as the sum
of dfnr and dfpr and remove these entries from the result
dictionary using Python's `dict.pop` method:

```python
report["mistreatment"] = abs(report["dfpr"]) + abs(report["dfnr"])
report.pop("dfpr")
report.pop("dfnr")
```

In addition to transforming results in this manner, reports
can also be reduced or combined alongside branches. Again,
these operations are applicable to all variable forks, 
but their usage is discouraged outside of report manipulation,
as they create new -and therefore potentially unforeseen- 
data branches.

To demonstrate reduction,
you can obtain the mean and maximum absolute value across branches
via the following code:

```python
mean_across_branches = fb.reduce(report, fb.mean, name="avg")
max_abs_across_branches = fb.reduce(report, fb.max, fb.abs)
```

The current supported reductions are *mean,max,min,sum*. 
Recuctions create new reports of only one branch.
The branch's name is dynamically derived by parameters 
(e.g., *"maxabs"*), but you can also use the `name` 
argument to set a specific name instead.

Reports, including reduced ones, can be combined alongside 
branches, as demonstrated in the following snippet:

```python
new_report = fb.comine(mean_across_branches, max_abs_across_branches)
```