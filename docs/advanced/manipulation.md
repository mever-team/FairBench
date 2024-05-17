# Report manipulation

!!! tip
    If you are trying to extract popular fairness definitions
    from reports, use [stamps](modelcards.md#stamps).

## Editing metrics

Reports are forks of dictionaries and you can use normal
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


## Reduction

Reports can be reduced alongside branches. Again,
this operation is in general applicable to all variable forks,
although this time usage is discouraged outside of 
report manipulation, as reduction creates new -and potentially 
unforeseen- data branches. That said, it is also the main mechanism
for summarizing multi-attribute reports into one measure.
Reduction internally runs three types of functions obtained
from its arguments:

- `transform` values found in the report for each metric, which can be either *None* or *abs*.
- `expand` the list of branch values for each metric, namely *None*, a pairwise *ratio* between values, or absolute *diff*erences between branch values.
- `reducer` method that takes a list of all branch values for each metric and summarizes them into one value. These can be *mean,max,min,sum,budget*, where the last one is the logarithm of the maximum declared in differential fairness formulations.

To demonstrate usage,
we next compute the mean and budget of the absolute value ratio.
Reduction creates new reports that comprise only one branch.
The branch's name is dynamically derived by parameters 
(e.g., *"budgetratioabs"*), but you can also set 
a specific name with the argument `name="ReductionName"`. 

```python
import fairbench as fb

sensitive = fb.Fork(case1=..., case2=...)
report = fb.accreport(predictions=..., labels=..., sensitive=sensitive)

mean_across_branches = fb.reduce(report, fb.mean, name="avg")
max_abs_across_branches = fb.reduce(report, fb.budget, expand=fb.ratio, transform=fb.abs)
```


!!! info
    Call `areduce` with the same arguments to 
    obtain a numeric output instead.

## Combining reports

Reports, including reductions, can be combined to
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

!!! tip
    Combine multiple reductions to create your own reports.

## Extracting from reports

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
