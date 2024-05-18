# Stamps

Stamps are popular fairness
definitions that can be extracted from 
[reports](../basics/reports.md)
and can be
used to create [fairness model cards](../advanced/modelcards.md).
You need stamps because they keep track of several metadata
that are shown in the model cards.
Create a stamp based on the following pattern:

```python
import fairbench as fb
report = ...  # code to generate a fairness report here
stamp = fb.stamps.stamp_name(report)
```

The same stamp may be supported by multiple reports,
in which case the report type changes the nuances of
the comparison. A database of all stamps is maintained in FairBench's 
repository [here](https://github.com/mever-team/FairBench/blob/main/stamps/common.yaml).
This database, as well as the rest of the library's
codebase is open to contributions.
Here we present all available stamps;
click on them to get a full description.

!!! tip
    Depending on the type of report, group comparisons may
    occur between groups or subgroups pairwise, for each 
    group against the total population, or for each group
    against its complement in the population. FairBench
    keeps track of this information in case its needed
    for model card descriptions or caveats. If multiple
    reports have been merged, one of the values is
    selected based on a priority order that each stamp
    dictates.

!!! danger
    Popular definitions reflected by stamps
    are not suited to all systems, and there may
    be definitions without a stamp.
    Some stamps may also only be used to expose certain biases
    but the absense of these biases may not necessarily
    imply fairness.
    For more details, look at the caveats and recommendations
    of produced model cards.


## Classification
The stamps presented here are often used to assess the 
fairness or bias of binary classifiers.

<button onclick="toggleCode('four_fifths', this)" class="toggle-reveal">
four_fifths</button>
<button onclick="toggleCode('accuracy', this)" class="toggle-reveal">
accuracy</button>
<button onclick="toggleCode('dfpr', this)" class="toggle-reveal">
dfpr</button>
<button onclick="toggleCode('dfnr', this)" class="toggle-reveal">
dfnr</button>

<div id="four_fifths" class="doc" markdown="span" style="display:none;">
The `four_fifths` stamp refers to the popular 4/5ths rule that infers discrimination
if positive rate ratios lies below 80%. We apply this for
all subgroups, an approach also known as differential fairness.
</div>

<div id="accuracy" class="doc" markdown="span" style="display:none;">
The `accuracy` stamp refers to the worst accuracy across all groups or subgroups.
</div>

<div id="dfpr" class="doc" markdown="span" style="display:none;">
The `dfpr` stamp refers to the difference in false positive rates between groups or subgroups.
We consider a multidimensional extension that, depending on considers the
worst difference.
</div>

<div id="dfnr" class="doc" markdown="span" style="display:none;">
The `dfnr` stamp refers to the difference in false negative rates between groups or subgroups.
We consider a multidimensional extension that considers the
worst difference.
</div>


## Ranking

The stamps presented here are often used to assess the 
fairness or bias of ranking when there is a binary outcome
of success against failure. This also models the case
of parity in the top-k predictions of recommender systems.

<button onclick="toggleCode('auc', this)" class="toggle-reveal">
auc</button>
<button onclick="toggleCode('abroca', this)" class="toggle-reveal">
abroca</button>
<button onclick="toggleCode('rbroca', this)" class="toggle-reveal">
rbroca</button>


<div id="auc" class="doc" markdown="span" style="display:none;">
The `auc` stamp refers to the worst AUC across all groups or subgroups.
</div>


<div id="abroca" class="doc" markdown="span" style="display:none;">
The `abroca` stamp refers to the absolute betweeness area between ROC curves (used to
compute AUC) of recommendation systems.
</div>

<div id="rbroca" class="doc" markdown="span" style="display:none;">
The `rbroca` stamp refers to the relative between area variation of abroca.
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