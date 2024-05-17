# Stamps

Here we describe the available stamps that can be used to
produce [fairness model cards](../advanced/modelcards.md).
A database of all stamps can be found in FairBench's 
repository [here](https://github.com/mever-team/FairBench/blob/main/stamps/common.yaml).
This database, as well as the rest of the library's
codebase is open to contributions at any time.

### `four_fifths_rule`
<div class="doc" markdown="span">
This is the popular 4/5ths rule that infers discrimination
if positive rate ratios lies below 80%. We apply this for
all subgroups, an approach also known as differential fairness.
</div>


### `accuracy`
<div class="doc" markdown="span">
This is the worst accuracy across all groups or subgroups.
</div>


### `dfpr`
<div class="doc" markdown="span">
The difference in false positive rates between groups or subgroups.
We consider a multidimensional extension that considers the
worst difference.
</div>

### `dfnr`
<div class="doc" markdown="span">
The difference in false negative rates between groups or subgroups.
We consider a multidimensional extension that considers the
worst difference.
</div>

### `abroca`
<div class="doc" markdown="span">
The absolute betweeness area between ROC curves (used to
compute AUC) of recommendation systems.
</div>

### `rbroca`
<div class="doc" markdown="span">
The relative between area variation of abroca.
</div>