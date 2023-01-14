# :checkered_flag: Variable forks

Variable forks declare different versions (branches) of variable
values, each of which should be treated differently.
For example, in multi-attribute fairness each fairness
attribute value can be a different branch.

## Fork definition

The easiest way to generate forks is by passing keyword
arguments to their class constructor, for instance per:

```python
import fairbench as fb

sensitive1, sensitive2 = ...
sensitive = fb.Fork(case1=sensitive1, case2=sensitive2)
```

Some or all branches of forks can be provided in 
the form of dictionaries passed as positional arguments.
For example, you can programmatically set their
names like this:
```python
case1name = "case1"
sensitive = fb.Fork({case1name: sensitive1}, case2=sensitive2)
```


## Working with multiple forked variables
If you have multiple forked variables,
they should all have the same branches.
Each branch will execute independently 
from the rest (some variable may be shared)
You can even create machine learning model branches:

```python
from sklearn.linear_model import LogisticRegression, MLPClassifier

x, y = ...

classifier = fb.Fork(case1=LogisticRegression(), case2=MLPClassifier())
classifier = classifier.fit(x, y)
yhat = classifier.predict(x)
```

Forks automatically try to call wrapped class methods,
so `classifier` fit is also a fork whose branches
hold the outcome of applying `fit` on each individual.
`x,y` could also have been forks, in which case the respective
value would have been passed to each branch.

Accessing branch values -if present- is done as class fields,
for example like `yhat = (yhat.case1+yhat.case2)/2`. This 
computation produces a factual tensor that is not
bound to any branch, but if was not there `yhat.case1`
and `yhat.case2` would be used during assessment of
case1 sensitive attribute values and case2 sensitive
attribute values. 

```python
print(classifier)
# case2: MLPClassifier()
# case1: LogisticRegression()
print(yhat)
# case2: [0 1 0 1 0 1 1 1]
# case1: [0 1 0 1 1 1 1 1]
print((yhat.case1+yhat.case2)/2)
# [0.  1.  0.  1.  0.5 1.  1.  1. ]
```

:warning: Accessing branch values or printing
forks when
[distributed computing](distributed.md)
is enabled waits until remote computations
conclude and retrieves data back.

Here is a visual view of how data 
are organized between branches:

![branches](branches.png)

:bulb: You can use branches to run several computations
pipelines concurrently.