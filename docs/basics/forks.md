# Forks

Forks declare variables that have one role
(e.g., they are sensitive attributes) but which exhibit
several named values, called branches.
When analysing fairness, branches correspond to
binary arrays of attribute values. For instance, there
may be separate branches of the sensitive attribute
indicating different genders or races for processed
data samples.

1. [Fork definition](#fork-definition)
2. [Intersectional analysis](#intersectional-analysis)

## Fork definition

Generate forks by passing keyword
arguments to a constructor. 
In the snippet below `men`, `women`, and `nonbiνary` are
branch names. Branch values can be anything,
though they will usually be arrays and internally
automatically converted to whatever backend
you are working with. You can provide any number of branches
with any names and access their values like object members,
as shown below.

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
sensitive = fb.Fork({"non-binary": np.array([0, 0, 0, 0, 1])}, 
                    men=np.array([1, 1, 0, 0, 0]), 
                    women=np.array([0, 0, 1, 1, 0]))
```

You can create forks by analysing categorical values found in iterables.
This can be done with the following pattern, which
creates two branches `genderMan,genderWoman` and stores binary
membership to each of those. 

```python
fork = fb.Fork(gender=fb.categories@["Man", "Woman", "Man", "Woman", "Nonbin"])
print(fork)
# genderMan: [1, 0, 1, 0, 0]
# genderWoman: [0, 1, 0, 1, 0]
# genderNonbin: [0, 0, 0, 0, 1]
```

You can add the outcomes of any number of 
category analyses to a fork. Do this via
positional arguments
(instead of named keyword arguments, such as `gender`)
to avoid prepending the keyword argument's name to branch names.
Any iterable can be analysed into categories instead of a list, 
including categorical tensors or arrays.


## Intersectional analysis

For more than one sensitive
attribute, add the branches you would declare for
every attribute more branches of the same fork.
For instance, the following is a valid fork that considers three
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
In this case, you might want to consider intersectional 
instead of more naive definitions of fairness
by creating all branch combinations with at least
one member per:

```python 
sensitive = sensitive.intersectional()
```
