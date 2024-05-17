# Metric comparisons

In FairBench [reports](../basics/reports.md) you will see 
several comparison mechanisms of [base metrics](metrics.md) that include expansions
and reductions. Here we summarize the mechanisms that
appear as column titles of the in-built report methods.

### `min`
<div class="doc" markdown="span">
Computes the minimum base metric value across all groups or
subgroups. Should have a high value to indicate that the base
metric has high value for even then less priviledged group.</div>

### `wmean`
<div class="doc" markdown="span">
Computes the average of base metric values across
groups and subgroups, weighted by group size. This
places greater emphasis to more populous groups
or subgroups. High values indicate high performance
across groups, and low values worse performance.
This mostly helps get sense of whether a `min`
comparison is substantially smaller than the average.
</div>

### `gini`
<div class="doc" markdown="span">
Computes the gini coefficient between metric values
across all groups or subgroups. Value of 0 indicates
that all groups achieve the same evaluation, whereas
value of 1 indicates full imbalance between metric
values.</div>

### `minratio`
<div class="doc" markdown="span">
The minimum ratio between base metric values
across groups and subgroups. The comparison is made
pairwise between each pair of groups. Values closer
to zero indicate imbalances between metric values
are large between at least one pair of groups, 
and value of 1 indicates perfect equality in
the measure assessments.
</div>

### `maxdiff`
<div class="doc" markdown="span">
The maximum difference between base metric values
across groups and subgroups. The comparison is made
pairwise between each pair of groups. Values closer
to zero indicate that all groups achieve the same
evaluation, and larger values indicate the worst
deviation between groups. Contrary to 
`minratio` or the `gini`
coefficient, this comparison has the same units 
as the measure being evaluated.
</div>

### `maxbarea`
<div class="doc" markdown="span">
The maximum betweeness area between the curves used to
compute metrics. In detail, if the base metric contains a
curve in its explanation, these curves are retrieved
and compared between pairs of groups (the metric's
value itself is not used). The comparison considers
the areas between the curves. The worst area across
pairs of groups is reported. Zero area difference
indicates that all groups have similar areas.
</div>


### `maxbdcg`
<div class="doc" markdown="span">
A similar strategy to `maxbarea` where differences
are made to matter more through a normalized discounted
comulative gain (NDCG) weighting. This is ideal for
comparing the top hit rates of recommendation systems,
but the formula has also been generalized to account
for ROC curves, where false positive rate differences
matter more when they occur for the same
small negative rates. Zero relative area difference again
indicates that all groups have similar ROC curves.
</div>

### `maxrarea`
<div class="doc" markdown="span">
A similar strategy to `maxbarea` where the comparison
between curve points is made not through an absolute
difference but how much their ratio deviates from 1.
Zero relative area difference again
indicates that all groups have similar ROC curves.
</div>
