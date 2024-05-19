# Metric comparisons

FairBench's main goal is to generate
fairness [reports](../basics/reports.md) that contain
many types of evaluations. Reports
look like the example below, where rows correspond to
[base metrics](metrics.md) computed for each sensitive
attribute dimension. Columns, on the other hand,
include expansions and reductions that summarize metric
comparisons across all protected groups or subgroups 
(for details on how to replicate this process with
custom metrics, see see [here](../advanced/manipulation.md)). 
Below we present the comparison mechanisms
computed by built-in report methods,
as well as the value at which
comparisons indicate full bias;
scan reports for particularly bad values to
notify stakeholders and explore why they occur.

```
--- Example multireport --- 
Metric          min             wmean           gini            minratio        maxdiff         maxbarea        maxrarea        maxbdcg        
auc             0.861           0.877           0.013           0.948           0.047           0.048           0.071           0.055          
avgscore        0.110           0.239           0.234           0.363           0.193           0.682           0.631           0.749          
tophr           0.667           0.778           0.100           0.667           0.333           nan             nan             nan            
toprec          0.001           0.002           0.392           0.121           0.004           nan             nan             nan            
avghr           0.389           0.592           0.220           0.389           0.611           0.611           0.611           0.696          
avgrepr         0.000           1.000           0.500           0.000           1.499           1.499           1.000           1.499   
```

## Performance

These report columns assess the performance of analysed 
systems while accounting explicitly for the existence of
subgroups.

<button onclick="toggleCode('min', this)" class="toggle-reveal">
min</button>
<button onclick="toggleCode('wmean', this)" class="toggle-reveal">
wmean</button>

<div id="min" class="doc" markdown="span" style="display:none;">
Computes the minimum base metric value across all groups or
subgroups. Should have a high value to indicate that the base
metric has high value for even then less priviledged group.

<br><em>Full bias: 0</em>
</div>

<div id="wmean" class="doc" markdown="span" style="display:none;">
Computes the average of base metric values across
groups and subgroups, weighted by group size. This
places greater emphasis to more populous groups
or subgroups. High values indicate high performance
across groups, and low values worse performance.
This mostly helps get sense of whether a `min`
comparison is substantially smaller than the average.

<br><em>Full bias: 0</em>
</div>

## Performance comparisons

These report columns directly compare the values of
base metrics between groups. Depending on the report
type, the groups being compared could be the pairs of
groups or subgroups you indicated in the sensitive
attribute fork, each group against the total population,
or each group against its complement in the population.

<button onclick="toggleCode('gini', this)" class="toggle-reveal">
gini</button>
<button onclick="toggleCode('minratio', this)" class="toggle-reveal">
minratio</button>
<button onclick="toggleCode('maxdiff', this)" class="toggle-reveal">
maxdiff</button>
<button onclick="toggleCode('minratio[vsAny]', this)" class="toggle-reveal">
minratio[vsAny]</button>
<button onclick="toggleCode('maxdiff[vsAny]', this)" class="toggle-reveal">
maxdiff[vsAny]</button>

<div id="gini" class="doc" markdown="span" style="display:none;">
Computes the gini coefficient between metric values
across all groups or subgroups. Value of 0 indicates
that all groups achieve the same evaluation, whereas
value of 1 indicates full imbalance between metric
values.

<br><em>Full bias: 1</em>
</div>

<div id="minratio" class="doc" markdown="span" style="display:none;">
The minimum ratio between base metric values
across groups and subgroups. The comparison is made
pairwise between each pair of groups. Values closer
to zero indicate that imbalances between 
at least one pair of groups are large, 
and value of 1 indicates perfect equality in
the measure assessments.

<br><em>Full bias: 0</em>
</div>

<div id="maxdiff" class="doc" markdown="span" style="display:none;">
The maximum difference between base metric values
across groups and subgroups. The comparison is made
pairwise between each pair of groups. Values closer
to zero indicate that all groups achieve the same
evaluation, and larger values indicate the worst
deviation between groups. This comparison has the same units 
as the measure being evaluated.

<br><em>Full bias: 1</em>
</div>

<div id="minratio[vsAny]" class="doc" markdown="span" style="display:none;">
The minimum ratio of base metric values
across groups and subgroups, when they are compared
to the total population. Values closer
to zero indicate that imbalances 
are large between at least one metric value
and the population as a whole
and value of 1 indicates that all groups
perfectly follow the whole population for the specific metric.

<br><em>Full bias: 0</em>
</div>


<div id="maxdiff[vsAny]" class="doc" markdown="span" style="display:none;">
The maximum difference of base metric values
across groups and subgroups, when they are compared
to the total population. Values closer
to zero indicate that all groups achieve the same evaluation
as the population as a whole, 
and larger values indicate the worst
deviation between a group and the population.
This comparison has the same units 
as the measure being evaluated.

<br><em>Full bias: 1</em>
</div>

## Curve comparisons

These report columns compare the underlying curves
of base measures. Which groups are compared is again
determined by the report type. The base measures should
keep track of curve explanations, otherwise this comparison
is not possible.

<button onclick="toggleCode('maxbarea', this)" class="toggle-reveal">
maxbarea</button>
<button onclick="toggleCode('maxbdcg', this)" class="toggle-reveal">
maxbdcg</button>
<button onclick="toggleCode('maxrarea', this)" class="toggle-reveal">
maxrarea</button>

<div id="maxbarea" class="doc" markdown="span" style="display:none;">
The maximum betweeness area between the curves used to
compute metrics. In detail, if the base metric contains a
curve in its explanation, these curves are retrieved
and compared between pairs of groups (the metric's
value itself is not used). The comparison considers
the areas between the curves. The worst area across
pairs of groups is reported. Zero area difference
indicates that all groups have similar areas.

<br><em>Full bias: 1</em>
</div>

<div id="maxbdcg" class="doc" markdown="span" style="display:none;">
A similar strategy to `maxbarea` where differences
are made to matter more through a normalized discounted
comulative gain (NDCG) weighting. This is ideal for
comparing the top hit rates of recommendation systems,
but the formula has also been generalized to account
for ROC curves, where false positive rate differences
matter more when they occur for the same
small negative rates. Zero relative area difference again
indicates that all groups have similar ROC curves.

<br><em>Full bias: 1</em>
</div>

<div id="maxrarea" class="doc" markdown="span" style="display:none;">
A similar strategy to `maxbarea` where the comparison
between curve points is made not through an absolute
difference but how much their ratio deviates from 1.
Zero relative area difference again
indicates that all groups have similar ROC curves.

<br><em>Full bias: 1</em>
</div>

<script>
function toggleCode(id, button) {
    var divsToHide = document.getElementsByClassName("doc");
    for(var i = 0; i < divsToHide.length; i++) {
        divsToHide[i].style.display = "none";
    }
    var codeBlock = document.getElementById(id);
    if (codeBlock.style.display === "none") {
        codeBlock.style.display = "block";
    } else {
        codeBlock.style.display = "none";
    }

    var buttons = document.getElementsByClassName("toggle-reveal");
    for (var j = 0; j < buttons.length; j++) {
        buttons[j].classList.remove("active");
    }
    button.classList.add("active");

}
</script>
