# FairBench
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Bringing together existing and new frameworks for fairness exploration.

**This project is in its pre-alpha phase.**

**Dependencies:** `numpy`,`eagerpy`

## :rocket: Quickstart

First, let's train a binary classification algorithm:

```python
import fairbench as fb
from sklearn.linear_model import LogisticRegression

x, y = ...

classifier = LogisticRegression()
classifier = classifier.fit(x, y)
yhat = classifier.predict(x)
```

Let us now consider some sensitive attribute. We can either use
a single `sensitive` array or consider multiple such attributes,
which we declare as a modal variable per:
```python
sensitive1, sensitive2 = ...
sensitive = fb.Modal(case1=sensitive1, case2=sensitive2)
```

After declaring the protected attribute, we generate a
report on the model's fairness and print it per:

```python
report = fb.report(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report)
```

```
                case1           case2          
accuracy        0.938           0.938           
dfnr            0.500           -0.167          
dfpr            0.071           -0.100          
prule           0.571           0.833    
```

Omitting some arguments from the report call will 
make it impossible to evaluate some measures, in
which case they will not be shown.

## :brain: Advanced usage
You might want to use different training and
prediction pipelines, depending on the

## :pencil: Contributing
`FairBench` was designed to be easily extensible.

<details>
<summary>Create a new metric.</summary>

1. Create it under `fairbench.metrics` module.
2. Add the `multimodal` decorator like this:
```
from faibench.model import multimodal

@multimodal
def metric(...):
    return ...
```
3. Reuse as many arguments found in other metrics as possible. 

:warning: Numeric inputs are automatically converted into 
[eagerpy](https://github.com/jonasrauber/eagerpy) tensors,
which use a functional interface to ensure interoperability.
You can use the `@multimodal_primitive` decorator to avoid
this conversion and work with specific primitives users of the
framework provide, but try not to do so. 

:bulb: If your metric should behave differently for different 
data modes, add a `mode=None` default argument in its
definition to get that mode.

</details>
