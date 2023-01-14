# :chart_with_upwards_trend: Reports

In the simplest case, you can generate 
fairness-aware reports by providing some
(preferably all) of the following arguments
of the `fairbench.report(...)` function:
* binary `predictions`
* ideal binary prediction `labels`
* binary `sensitive` attribute

You can use variable [forks](branches.md)
to handle multi-attribute values.
:warning: For the time being, you need
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