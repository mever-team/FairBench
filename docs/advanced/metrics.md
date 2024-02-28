# Performance metrics

FairBench [reports](../basics/reports.md) 
implement several definitions of fairness that
quantify imbalances between groups of people (e.g.,
different genders) in terms of them obtaining different 
assessments by base performance metrics. These assessments 
are often further 
[reduced](manipulation.md#reduction) across groups
of samples with different sensitive attribute values.

Here, we present base metrics used to assess AI
that reports use. All metrics computed
on a subset of 'sensitive samples', which form
the group being examined each time. Outputs are
wrapped into explainable objects that keep track of
relevant metadata.

1. [Classification](#classification)
2. [Ranking](#ranking)
3. [Regression](#regression)


## Classification
Classification metrics assess binary predictions. Unless stated
otherwise, the following arguments need to be provided:


| Argument    | Role                | Values                                                           |
|-------------|---------------------|------------------------------------------------------------------|
| predictions | system output       | binary array                                                     |      
| labels      | prediction target   | binary array                                                     | 
| sensitive   | sensitive attribute | fork of arrays with elements in $[0,1]$ (either binary or fuzzy) |



### `accuracy`
<div class="doc" markdown="span">
Computes the accuracy for correctly predicting provided binary labels
for sensitive data samples. Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, true predictions
</em></div></div>


### `pr`
<div class="doc" markdown="span">
Computes the positive rate of binary predictions for sensitive
data samples. Returns a float in the range $[0,1]$. This metric 
does not use the `labels` argument.

<br><div class="explain"><em>Explanation: 
number of samples, positive predictions
</em></div></div>


### `positives`
<div class="doc" markdown="span">
Computes the number of positive predictions for
sensitive data samples. Returns a float in the range $[0,\infty)$. 
This metric does not use the `labels` argument.

<br><div class="explain"><em>Explanation: 
number of samples
</em></div></div>


### `tpr`
<div class="doc" markdown="span">
Computes the true positive rate of binary predictions for sensitive
data samples. Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, number of positives, number of true positives
</em></div></div>


### `tnr`
<div class="doc" markdown="span">
Computes the true negative rate of binary predictions for sensitive
data samples. Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, number of negatives, number of true negatives
</em></div></div>


### `fpr`
<div class="doc" markdown="span">
Computes the false positive rate of binary predictions for sensitive
data samples. Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, number of positives, number of false positives
</em></div></div>


### `fnr`
<div class="doc" markdown="span">
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


### `auc`
<div class="doc" markdown="span">
Computes the area under curve of the receiver operating
characteristics for sensitive data samples.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, the receiver operating characteristic curve
</em></div></div>


### `phi`
<div class="doc" markdown="span">
Computes the score mass of
sensitive data samples compared 
to the total scores.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, sensitive scores
</em></div></div>


### `tophr`
<div class="doc" markdown="span">
Computes the hit rate, i.e., precision, for a set number of
top scores for sensitive data samples. This is
used to assess recommendation systems. By default, the
top-3 hit rate is analysed.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, top scores, true top scores
</em></div></div>

| Optional argument | Role      | Values                            |
|---------------------|-----------|-----------------------------------|
| top                 | parameter | integer in the range $[1,\infty)$ |


### `toprec`
<div class="doc" markdown="span">
Computes the recall for a set number of
top scores for sensitive data samples. This is
used to assess recommendation systems. By default, the
top-3 recall is analysed.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, top scores, true top scores
</em></div></div>

| Optional argument | Role      | Values                            |
|---------------------|-----------|-----------------------------------|
| top                 | parameter | integer in the range $[1,\infty)$ |


### `topf1`
<div class="doc" markdown="span">
Computes the f1-score for a set number of
top scores for sensitive data samples. This is
the harmonic mean between hr and preck and is
used to assess recommendation systems. By default, the
top-3 f1 is analysed.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, top scores, true top scores
</em></div></div>

| Optional argument | Role      | Values                            |
|---------------------|-----------|-----------------------------------|
| top                 | parameter | integer in the range $[1,\infty)$ |




### `tophr`
<div class="doc" markdown="span">
Computes the average hit rate/precession 
across different numbers of top scores
with correct predictions.  By default, the
top-3 average precision is computed.
Returns a float in the range $[0,1]$.

<br><div class="explain"><em>Explanation: 
number of samples, top scores, hr curve
</em></div></div>

| Optional argument | Role      | Values                            |
|---------------------|-----------|-----------------------------------|
| top                 | parameter | integer in the range $[1,\infty)$ |




## Regression

Regression metrics assess scores that aim to reproduce
desired target scores. The following arguments need to be provided:


| Argument  | Role                | Values                                                           |
|-----------|---------------------|------------------------------------------------------------------|
| scores    | system output       | any float array                                                  |      
| targets   | prediction target   | any float array                                                  | 
| sensitive | sensitive attribute | fork of arrays with elements in $[0,1]$ (either binary or fuzzy) |


### `max_error`
<div class="doc" markdown="span">
Computes the maximum absolute error between scores and targets
for sensitive data samples. Returns a float in the range $[0,\infty)$.

<br><div class="explain"><em>Explanation: ---
</em></div></div>

### `mae`
<div class="doc" markdown="span">
Computes the mean of the absolute error between scores and targets
for sensitive data samples. Returns a float in the range $[0,\infty)$.

<br><div class="explain"><em>Explanation:
number of samples, sum of absolute errors
</em></div></div>

### `mse`
<div class="doc" markdown="span">
Computes the mean of the square error between scores and targets
for sensitive data samples. Returns a float in the range $[0,\infty)$.

<br><div class="explain"><em>Explanation:
number of samples, sum of square errors
</em></div></div>

### `rmse`
<div class="doc" markdown="span">
Computes the root of mse.
Returns a float in the range $[0,\infty)$.

<br><div class="explain"><em>Explanation:
number of samples, sum of square errors
</em></div></div>

### `r2`
<div class="doc" markdown="span">
Computes the r2 score between scores
and target values, adjusted for the 
provided degree of freedom (default is zero).
Returns a float in the range $(-\infty,1]$,
where larger values correspond to better
estimation and models that output the mean
are evaluated to zero.

<br><div class="explain"><em>Explanation:
number of samples, sum of square errors, degrees of freedom
</em></div></div>

| Optional argument | Role      | Values                            |
|---------------------|-----------|-----------------------------------|
| deg_freedom         | parameter | integer in the range $[0,\infty)$ |      


### `pinball`
<div class="doc" markdown="span">
Computes the pinball deviation between scores
and target values for a balance parameter alpha
(default is 0.5).
Returns a float in the range $[0,\infty)$,
where smaller values correspond to better
estimation.

<br><div class="explain"><em>Explanation:
number of samples
</em></div></div>

| Optional argument | Role      | Values                     |
|---------------------|-----------|----------------------------|
| alpha               | parameter | float in the range $[0,1]$ |      
