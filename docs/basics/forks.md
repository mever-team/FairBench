# Forks

Forks declare variables that have one role
(e.g., they are sensitive attributes) but which can exhibit
different values in different named scenarios, called branches.
When analysing fairness, each branch corresponds to an
array corresponding to some attribute value - see below for examples.

1. [Fork definition](#fork-definition)
2. [Intersectional analysis](#intersectional-analysis)

## Fork definition

Generate forks by passing keyword
arguments to a constructor. 
In the snippet below `men`, `women`, and `nonbiνary` are
fork branch names. Branch values can also be anything,
though they will usually be arrays of whatever backend
you are working with. You can provide any number of branches
with any names and access them like object members.

```python
import fairbench as fb
import numpy as np

sensitive = fb.Fork(men=np.array([1, 1, 0, 0, 0]), 
                    women=np.array([0, 0, 1, 1, 0]),
                    nonbinary=np.array([0, 0, 0, 0, 1]))
print(sensitive.nonbinary)
# [0, 0, 0, 0, 1]
```



You can also set some (or all) branches programmatically
by passing a dictionary as a positional argument like so:
```python
case1name = "case1"
sensitive = fb.Fork({"non-binary": np.array([0, 0, 0, 0, 1])}, 
                    men=np.array([1, 1, 0, 0, 0]), 
                    women=np.array([0, 0, 1, 1, 0]))
```

To fork generation in specific scenarios, 
you can analyse categorical values found in iterables.
This can be done with the following pattern, which
creates two branches `genderMan,genderWoman` and stores binary
membership to each of those:

```python
fork = fb.Fork(gender=fb.categories@["Man", "Woman", "Man", "Woman", "Nonbin"])
print(fork)
# genderMan: [1, 0, 1, 0, 0]
# genderWoman: [0, 1, 0, 1, 0]
# genderNonbin: [0, 0, 0, 0, 1]
```

You can input the outcome of any number of 
category analyses. Add them as positional (instead of named keyword)
arguments to avoid prepending a keyword (e.g., `gender`) 
to respective branch names.
Any iterable can be analysed into categories instead of a list, 
including categorical tensors or arrays.


## Intersectional analysis

For more than one sensitive
attribute, add the branches you would declare for
every attribute more branches of the same fork.
For instance, this is a valid fork that considers three
gender attribute values and one binary sensitive attribute 
value for young vs old people:

```python
import numpy as np

sensitive = fb.Fork(fb.categories@["Man", "Woman", "Man", "Woman", "Nonbin"], 
                    IsOld=np.array([0, 1, 0, 1, 0]))
# Man: [1, 0, 1, 0, 0]
# Woman: [0, 1, 0, 1, 0]
# Nonbin: [0, 0, 0, 0, 1]
# IsOld0.0 [1, 0, 0, 0, 1]
# IsOld1.0 [0, 1, 1, 1, 0]
```

For more than one sensitive attributes,
branches capturing the values of different attributes
will have overlapping non-zeroes.
In this case, you might want to consider intersectional fairness
instead more naive definitions of fairness
by creating all branch combinations with at least
one member per:

```python 
sensitive = sensitive.intersectional()
```
