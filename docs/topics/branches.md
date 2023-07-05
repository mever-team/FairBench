# Forks and branches

Forks declare different versions (branches) of variable
values, each of which should be treated differently.
For example, for multivalued sensitive attributes,
each value should correspond to a different branch that stores a binary
array indicating which samples exhibit the corresponding value.

1. [Fork definition](#fork-definition)
2. [Multivalue multiattribute considerations](#multivalue-multiattribute-considerations)
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
want. For multivalued sensitive attributes, each branch
should capture binary arrays
indicating membership of respective data samples to respective
groups of people.

To have programmatically-defined names,
some or all branches of forks can be provided in 
the form of dictionaries passed as positional arguments.
For example, you can set a custom name for the first branch like
this:
```python
case1name = "case1"
sensitive = fb.Fork({case1name: sensitive1}, case2=sensitive2)
```

To simplify definition of forks as binary arrays, 
you can analyse categorical values found in iterables.
This can be done with the following pattern, which
creates two branches `Man,Woman` and stores binary
membership to each of those:

```python
fork fb,Fork(fb.categories@["Man", "Woman", "Man", "Woman", "Nonbin"])
print(fork)
# Man: [1, 0, 1, 0, 0]
# Woman: [0, 1, 0, 1, 0]
# Nonbin: [0, 0, 0, 0, 1]
```

If you want to prepend a prefix, add the categorical
analysis as a keyword argument with the keyword being
the prefix:

```python
fork = fb.Fork(gender=fb.categories@["Man", "Woman", "Man", "Woman", "Nonbin"])
print(fork)
# genderMan: [1, 0, 1, 0, 0]
# genderWoman: [0, 1, 0, 1, 0]
# genderNonbin: [0, 0, 0, 0, 1]
```

Any iterable can be analysed, including arrays:

```python
men = np.array([1, 0, 1, 0])
fork = fb.Fork(gender=fb.categories@men)
print(fork)
# gender1.0: [1, 0, 1, 0]
# gender0.0: [0, 1, 0, 1]
```


## Multivalue multiattribute considerations
If you have more than one sensitive
attribute, just add the branches you would declare for
every attribute more branches of the same fork.
For instance, this is a valid fork that considers three
gender attribute values and one binary sensitive attribute 
value for old people:

```python
import numpy as np

sensitive = fb.Fork(Men=np.array([1, 1, 0, 0, 0]), 
                    Women=np.array([0, 0, 1, 1, 0]), 
                    Nonbin=np.array([0, 0, 0, 0, 1]), 
                    IsOld=np.array([0, 1, 0, 1, 0]))
```

You can use keyword arguments over categorical analysis
to add prefixes, as shown below:

```python
import numpy as np

sensitive = fb.Fork(gender=fb.categories@np.array([1, 1, 0, 0, 0]), 
                    race=fb.categories@np.array([0, 1, 1, 1, 0]))
print(sensitive)
# gender0.0 [0, 0, 1, 1, 1]
# gender1.0 [1, 1, 0, 0, 0]
# race0.0 [1, 0, 0, 0, 1]
# race1.0 [0, 1, 1, 1, 0]
```

In these cases where you have more than one attributes,
branches stemming capturing values of different attributes
will have overlapping non-zeroes.
In this case, you might want to consider intersectional fairness
instead more naive definitions of fairness
by creating all branch combinations with at least
one member per:

```python 
sensitive = sensitive.intersectional()
```


## Working with multiple forked variables
*These are advanced capabilities that are not needed to generate fairness reports.*

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