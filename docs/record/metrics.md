# Base metrics

FairBench [reports](../basics/reports.md) 
implement several definitions of fairness that
quantify imbalances between groups of people (e.g.,
different genders) in terms of them obtaining different 
assessments by base performance metrics. These assessments 
are often further 
[reduced](../advanced/manipulation.md#reduction) across groups
of samples with different sensitive attribute values.
Here, we present base metrics used to assess AI
that reports use. All metrics computed
on a subset of 'sensitive samples', which form
the group being examined each time. Outputs are
wrapped into explainable objects that keep track of
relevant metadata.

## Classification
Classification metrics assess binary predictions. Unless stated
otherwise, the following arguments need to be provided:

| Argument    | Role                | Values                                                           |
|-------------|---------------------|------------------------------------------------------------------|
| predictions | system output       | binary array                                                     |      
| labels      | prediction target   | binary array                                                     | 
| sensitive   | sensitive attribute | fork of arrays with elements in $[0,1]$ (either binary or fuzzy) |

<button onclick="toggleCode('accuracy', this)" class="toggle-reveal">
accuracy</button>
<button onclick="toggleCode('pr', this)" class="toggle-reveal">
pr</button>
<button onclick="toggleCode('positives', this)" class="toggle-reveal">
positives</button>
<button onclick="toggleCode('tpr', this)" class="toggle-reveal">
tpr</button>
<button onclick="toggleCode('tnr', this)" class="toggle-reveal">
tnr</button>
<button onclick="toggleCode('fpr', this)" class="toggle-reveal">
fpr</button>
<button onclick="toggleCode('fnr', this)" class="toggle-reveal">
fnr</button>

<div id="accuracy" class="doc" markdown="span" style="display:none;">
Computes the accuracy for correctly predicting provided binary labels
for sensitive data samples. Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, true predictions
</em></div></div>

<div id="pr" class="doc" markdown="span" style="display:none;">
Computes the positive rate of binary predictions for sensitive
data samples. Returns a float in the range $[0,1]$. This metric 
does not use the `labels` argument.

<br><div class="explain"><em>Explanation: 
number of samples, positive predictions
</em></div></div>

<div id="positives" class="doc" markdown="span" style="display:none;">
Computes the number of positive predictions for
sensitive data samples. Returns a float in the range $[0,\infty)$. 
This metric does not use the `labels` argument.

<br><div class="explain"><em>Explanation: 
number of samples
</em></div></div>

<div id="tpr" class="doc" markdown="span" style="display:none;">
Computes the true positive rate of binary predictions for sensitive
data samples. Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, number of positives, number of true positives
</em></div></div>

<div id="tnr" class="doc" markdown="span" style="display:none;">
Computes the true negative rate of binary predictions for sensitive
data samples. Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, number of negatives, number of true negatives
</em></div></div>

<div id="fpr" class="doc" markdown="span" style="display:none;">
Computes the false positive rate of binary predictions for sensitive
data samples. Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, number of positives, number of false positives
</em></div></div>

<div id="fnr" class="doc" markdown="span" style="display:none;">
Computes the false negative rate of binary predictions for sensitive
data samples. Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, number of negatives, number of false negatives
</em></div></div>

## Ranking

Ranking metrics assess scores that aim to approach
provided labels. The following arguments need to be provided:

| Argument  | Role                | Values                                                           |
|-----------|---------------------|------------------------------------------------------------------|
| scores    | system output       | array with elements in $[0,1]$                                   |      
| labels    | prediction target   | binary array                                                     | 
| sensitive | sensitive attribute | fork of arrays with elements in $[0,1]$ (either binary or fuzzy) |

<button onclick="toggleCode('auc', this)" class="toggle-reveal">
auc</button>
<button onclick="toggleCode('phi', this)" class="toggle-reveal">
phi</button>
<button onclick="toggleCode('tophr', this)" class="toggle-reveal">
tophr</button>
<button onclick="toggleCode('toprec', this)" class="toggle-reveal">
toprec</button>
<button onclick="toggleCode('topf1', this)" class="toggle-reveal">
topf1</button>
<button onclick="toggleCode('tophr_avg', this)" class="toggle-reveal">
tophr_avg</button>

<div id="auc" class="doc" markdown="span" style="display:none;">
Computes the area under curve of the receiver operating
characteristics for sensitive data samples.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, the receiver operating characteristic curve
</em></div></div>

<div id="phi" class="doc" markdown="span" style="display:none;">
Computes the score mass of
sensitive data samples compared 
to the total scores.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, sensitive scores
</em></div></div>

<div id="tophr" class="doc" markdown="span" style="display:none;">
Computes the hit rate, i.e., precision, for a set number of
top scores for sensitive data samples. This is
used to assess recommendation systems. By default, the
top-3 hit rate is analysed.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, top scores, true top scores
</em></div>

| Optional argument | Role      | Values                            |
|---------------------|-----------|-----------------------------------|
| top                 | parameter | integer in the range $[1,\infty)$ |

</div>

<div id="toprec" class="doc" markdown="span" style="display:none;">
Computes the recall for a set number of
top scores for sensitive data samples. This is
used to assess recommendation systems. By default, the
top-3 recall is analysed.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, top scores, true top scores
</em></div>

| Optional argument | Role      | Values                            |
|---------------------|-----------|-----------------------------------|
| top                 | parameter | integer in the range $[1,\infty)$ |

</div>

<div id="topf1" class="doc" markdown="span" style="display:none;">
Computes the f1-score for a set number of
top scores for sensitive data samples. This is
the harmonic mean between hr and preck and is
used to assess recommendation systems. By default, the
top-3 f1 is analysed.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, top scores, true top scores
</em></div>

| Optional argument | Role      | Values                            |
|---------------------|-----------|-----------------------------------|
| top                 | parameter | integer in the range $[1,\infty)$ |

</div>

<div id="tophr_avg" class="doc" markdown="span" style="display:none;">
Computes the average hit rate/precession 
across different numbers of top scores
with correct predictions.  By default, the
top-3 average precision is computed.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, top scores, hr curve
</em></div>

| Optional argument | Role      | Values                            |
|---------------------|-----------|-----------------------------------|
| top                 | parameter | integer in the range $[1,\infty)$ |

</div>

## Regression

Regression metrics assess scores that aim to reproduce
desired target scores. The following arguments need to be provided:

| Argument  | Role                | Values                                                           |
|-----------|---------------------|------------------------------------------------------------------|
| scores    | system output       | any float array                                                  |      
| targets   | prediction target   | any float array                                                  | 
| sensitive | sensitive attribute | fork of arrays with elements in $[0,1]$ (either binary or fuzzy) |

<button onclick="toggleCode('max_error', this)" class="toggle-reveal">
max_error</button>
<button onclick="toggleCode('mae', this)" class="toggle-reveal">
mae</button>
<button onclick="toggleCode('mse', this)" class="toggle-reveal">
mse</button>
<button onclick="toggleCode('rmse', this)" class="toggle-reveal">
rmse</button>
<button onclick="toggleCode('r2', this)" class="toggle-reveal">
r2</button>
<button onclick="toggleCode('pinball', this)" class="toggle-reveal">
pinball</button>

<div id="max_error" class="doc" markdown="span" style="display:none;">
Computes the maximum absolute error between scores and targets
for sensitive data samples. Returns a float in the range $[0,\infty)$.

<br><div class="explain"><em>Explanation: ---
</em></div></div>

<div id="mae" class="doc" markdown="span" style="display:none;">
Computes the mean of the absolute error between scores and targets
for sensitive data samples. Returns a float in the range $[0,\infty)$.

<br><div class="explain"><em>Explanation:
number of samples, sum of absolute errors
</em></div></div>

<div id="mse" class="doc" markdown="span" style="display:none;">
Computes the mean of the square error between scores and targets
for sensitive data samples. Returns a float in the range $[0,\infty)$.

<br><div class="explain"><em>Explanation:
number of samples, sum of square errors
</em></div></div>

<div id="rmse" class="doc" markdown="span" style="display:none;">
Computes the root of mse.
Returns a float in the range $[0,\infty)$.

<br><div class="explain"><em>Explanation:
number of samples, sum of square errors
</em></div></div>

<div id="r2" class="doc" markdown="span" style="display:none;">
Computes the r2 score between scores
and target values, adjusted for the 
provided degree of freedom (default is zero).
Returns a float in the range $(-\infty,1]$,
where larger values correspond to better
estimation and models that output the mean
are evaluated to zero.

<br><div class="explain"><em>Explanation:
number of samples, sum of square errors, degrees of freedom
</em></div>

| Optional argument | Role      | Values                            |
|---------------------|-----------|-----------------------------------|
| deg_freedom         | parameter | integer in the range $[0,\infty)$ |      

</div>

<div id="pinball" class="doc" markdown="span" style="display:none;">
Computes the pinball deviation between scores
and target values for a balance parameter alpha
(default is 0.5).
Returns a float in the range $[0,\infty)$,
where smaller values correspond to better
estimation.

<br><div class="explain"><em>Explanation:
number of samples
</em></div>

| Optional argument | Role      | Values                     |
|---------------------|-----------|----------------------------|
| alpha               | parameter | float in the range $[0,1]$ |      

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
