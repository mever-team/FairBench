# Computational branches

If you have multiple [forks](../basics/forks.md),
they should all have the same branches.
Each branch will execute independently 
of the rest (non-fork inputs are shared between them).
You can even create machine learning model 
forks, where a different model is applied 
on different branches:

```python
from sklearn.linear_model import LogisticRegression, MLPClassifier

x, y = ...

classifier = fb.Fork(case1=LogisticRegression(), case2=MLPClassifier())
classifier = classifier.fit(x, y)
yhat = classifier.predict(x)
```

Forks automatically try to call wrapped class methods,
i.e., `classifier.fit` is also a fork whose branches
hold the outcome of applying `fit` on each branch's model.
The inputs `x,y` could also have been forks, 
in which case each branch would have been trained on
respective values.

Recall that branch values can be accessed via class fields,
for example like `yhat = (yhat.case1+yhat.case2)/2`. This 
computation produces a factual value that is not
bound to any branch. On the other hand `yhat.case1`
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

A visual view of how data 
are organized across branches follows. Some
variables are identical but others
obtain different values per branch. The same
code is run on all branches concurrently and
independently.

![branches](branches.png)

!!! tip
    Use branches to run several computation pipelines concurrently.

!!! danger 
    Avoid overlapping names between branches 
    and class fields or methods, as they are both 
    accessed with the same annotation.
    If there is confusion, branch values will be obtained.
