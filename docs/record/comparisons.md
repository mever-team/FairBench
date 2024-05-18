# Metric comparisons

In FairBench [reports](../basics/reports.md) you will see 
several comparison mechanisms of [base metrics](metrics.md) that include expansions
and reductions. Here we summarize the mechanisms that
appear as column titles of the in-built report methods.

## Worst performance

These report columns take the worst value of base metrics
across groups.

<button onclick="toggleCode('min', this)" class="toggle-reveal">
min</button>
<button onclick="toggleCode('wmean', this)" class="toggle-reveal">
wmean</button>
<button onclick="toggleCode('gini', this)" class="toggle-reveal">
gini</button>

<div id="min" class="doc" markdown="span" style="display:none;">
Computes the minimum base metric value across all groups or
subgroups. Should have a high value to indicate that the base
metric has high value for even then less priviledged group.
</div>

<div id="wmean" class="doc" markdown="span" style="display:none;">
Computes the average of base metric values across
groups and subgroups, weighted by group size. This
places greater emphasis to more populous groups
or subgroups. High values indicate high performance
across groups, and low values worse performance.
This mostly helps get sense of whether a `min`
comparison is substantially smaller than the average.
</div>

<div id="gini" class="doc" markdown="span" style="display:none;">
Computes the gini coefficient between metric values
across all groups or subgroups. Value of 0 indicates
that all groups achieve the same evaluation, whereas
value of 1 indicates full imbalance between metric
values.
</div>

## Performance comparisons

These report columns directly compare the values of
base metrics between groups. Depending on the report
type, the groups being compared could be the pairs of
groups or subgroups you indicated in the sensitive
attribute fork, each group against the total population,
or each group against its complement in the population.

<button onclick="toggleCode('minratio', this)" class="toggle-reveal">
minratio</button>
<button onclick="toggleCode('maxdiff', this)" class="toggle-reveal">
maxdiff</button>

<div id="minratio" class="doc" markdown="span" style="display:none;">
The minimum ratio between base metric values
across groups and subgroups. The comparison is made
pairwise between each pair of groups. Values closer
to zero indicate imbalances between metric values
are large between at least one pair of groups, 
and value of 1 indicates perfect equality in
the measure assessments.
</div>

<div id="maxdiff" class="doc" markdown="span" style="display:none;">
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
</div>

<div id="maxrarea" class="doc" markdown="span" style="display:none;">
A similar strategy to `maxbarea` where the comparison
between curve points is made not through an absolute
difference but how much their ratio deviates from 1.
Zero relative area difference again
indicates that all groups have similar ROC curves.
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
