# :checkered_flag: Variable forks

Variable forks declare different versions (branches) of variable
values, each of which should be treated differently.
For example, for multi-valued sensitive attributes,
each value has a different branch that stores a binary
array indicating which samples exhibit the corresponding value.

1. [Fork definition](#fork-definition)
3. [Working with multiple forked variables](#working-with-multiple-forked-variables)

## Fork definition

The easiest way to generate forks is via passing keyword
arguments to their class constructor, for instance per:

```python
import fairbench as fb

sensitive1, sensitive2 = ...
sensitive = fb.Fork(case1=sensitive1, case2=sensitive2)
```

The branch names `case1`, `case2`, etc can be anything you 
want. You can also have more than two branches that consider
multiple attribute values. If you have more than one sensitive
attribute, just add the branches you would declare for
every attribute as a fork branch.
For instance, this is a valid fork that considers three
gender attribute values and one binary sensitive attribute 
value:

```python
import numpy as np

sensitive = fb.Fork(Men=
                    np.array([1, 1, 0, 0, 0]), 
                    Women=np.array([0, 0, 1, 1, 0]), 
                    Nonbin=np.array([0, 0, 0, 0, 1]), 
                    IsOld=np.array([0, 1, 0, 1, 0]))
```

If you have more than one sensitive attribute,
fork branches that correspond to values of different attributes
will have overlapping non-zeroes.
In this case, you might want to consider intersectional fairness
by creating all branch combinations with at least
one member per:

```python 
sensitive = sensitive.intersectional()
```

Some or all branches of forks can be provided in 
the form of dictionaries passed as positional arguments.
For example, you can programmatically set their
names like this:
```python
case1name = "case1"
sensitive = fb.Fork({case1name: sensitive1}, case2=sensitive2)
```

You can also define forks of binary arrays 
by analysing categorical values found in iterables.
This can be done with the following pattern, which
creates two branches `Man,Woman` and stores binary
membership to each of those:

```python
fork fb,Fork(fb.categories@["Man", "Woman", "Man", "Woman"])
print(fork)
# Man: [1, 0, 1, 0]
# Woman: [0, 1, 0, 1]
```

If you want to prepend a prefix, add this categorical
analysis as a keyword argument with the keyword being
the prefix:

```python
fork = fb.Fork(gender=fb.categories@["Man", "Woman", "Man", "Woman"])
print(fork)
# genderMan: [1, 0, 1, 0]
# genderWoman: [0, 1, 0, 1]
```

Any Python iterable can be analysed, including arrays:

```python
men = np.array([1, 0, 1, 0])
fork = fb.Fork(gender=fb.categories@men)
print(fork)
# gender1.0: [1, 0, 1, 0]
# gender0.0: [0, 1, 0, 1]
```





## Working with multiple forked variables
*These are advanced capabilities that are not needed to generate fairness report.*

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