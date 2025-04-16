from playground.v1.basics.test import sensitive

# Dimensions

`Dimensions` is a flexible data structures that stores 
multidimensional values, such as
sensitive attributes or predictions and labels for
multiple classes. Each dimension holds its own
value. For example, you might have a 
dimension for each gender; those dimensions would be 
binary arrays indicating the presence of the respective
attribute in data samples.

As a common use case is to
define a multidimensional sensitive attribute 
for intersectional fairness, like so:

```python
import fairbench as fb

gender = ["Male", "Male", "Female", "Female", "Nonbinary"]  # any iterable
race = ["Black", "White", "White", "Black", "White"]  # any iterable
sensitive = fb.Dimensions(fb.categories @ gender, fb.categories @ race) 
sensitive = sensitive.intersectional() 
print(sensitive)
```

```text
Black: [1 0 0 1 0]
White: [0 1 1 0 1]
Female: [0 0 1 1 0]
Female&Black: [0 0 0 1 0]
Female&White: [0 0 1 0 0]
Male: [1 1 0 0 0]
Male&Black: [1 0 0 0 0]
Male&White: [0 1 0 0 0]
Nonbinary: [0 0 0 0 1]
Nonbinary&White: [0 0 0 0 1]
```


!!! info
    FairBench accepts a wide range of data types as dimension values.
    These types include lists, arrays, and tensors of popular machine learning frameworks.


## Explicit dimensions
For values that are non-dictionaries,
keywords like `men`, `women`, and `nonbinary` are dimension names. 
Dimension values will usually be lists,
numpy arrays or deep learning tensors. 
Provide any number of dimensions
with any names and access their values 
as either members of the `Dimensions` object
or dictionary values.

```python
import fairbench as fb

sensitive = fb.Dimensions(
    men=[1, 1, 0, 0, 0], 
    women=[0, 0, 1, 1, 0],
    nonbinary=[0, 0, 0, 0, 1]
)
print(sensitive.nonbinary)
print(sensitive["nonbinary"]) # does the same
```

```text
[0, 0, 0, 0, 1]
[0, 0, 0, 0, 1]
```

## From dictionaries

To set dimensions names programmatically or use names 
with special characters, such as spaces, pass as a positional argument
to the constructor a dictionary 
that maps names to values. 
Do this with any number of dictionaries, and in addition to branches 
declared via keyword arguments.

```python
import fairbench as fb

sensitive = fb.Dimensions(
    {"non-binary": [0, 0, 0, 0, 1]}, 
    men=[1, 1, 0, 0, 0], 
    women=[0, 0, 1, 1, 0]
)
print(sensitive.men)
print(sensitive["men"])  # does the same
```

```text
[1, 1, 0, 0, 0]
[1, 1, 0, 0, 0]
```

It is acceptable to have dimensions with different nature, 
such as multiple sensitive attributes and attribute values.
Treat each value of each dimension as a separate dimension.
For example, mix in gender and age sensitive attributes
in the same `Dimensions` object.

```python
import fairbench as fb

sensitive = fb.Dimensions(
    men=[1, 1, 0, 0, 0],
    nonmen=[0, 0, 1, 1, 1],
    IsOld=[0, 1, 0, 1, 0]
)
```

To add multiple sensitive attributes without worrying about 
conflicting branch names, pass dictionaries as keyword 
arguments. This prepends the keyword argument name to 
all generated branch names.

```python
import fairbench as fb

sensitive = fb.Dimensions(
    gender={"1": [0, 0, 1, 1, 0], "0": [1, 1, 0, 0, 0], "?": [0, 0, 0, 0, 1]},
    isold={"1": [0, 1, 0, 1, 0], "0": [1, 0, 1, 0, 1]}
)
```



## Unpack to dictionaries

Here are helper operators that can convert iterable
data into dictionaries to pass into `Dimensions`.
Create multiple dimensions by analyzing categorical 
values found in iterables with the
`categories@` operator.
For example, when applied on a list whose entries are among
"Man". "Woman", "Nonbin"
this operator creates three branches storing binary membership for each
of those declared genders.

```python
import fairbench as fb

sensitive = fb.Dimensions(fb.categories@["Man", "Woman", "Man", "Woman", "Nonbin"])
print(sensitive)
```
```text
genderMan: [1, 0, 1, 0, 0]
genderWoman: [0, 1, 0, 1, 0]
genderNonbin: [0, 0, 0, 0, 1]
```

Add the outcomes of multiple category analyses 
to `Dimensions`; use named keyword arguments to prepend the 
that name to dimensions names, or put all category analyses
as positional arguments to just merge their branches.
Any Python iterable can be analyzed into categories.
This includes lists, pandas dataframe
columns, categorical tensors, or numpy arrays.

```python
import fairbench as fb

gender = ...  # iterable (e.g., list) of gender attribute for each data sample
race = ...  # iterable (e.g., list) of race attribute for each data sample
sensitive = fb.Dimensions(
    gender=fb.categories @ gender, 
    race=fb.categories @ race
) 
```

Use the `fuzzy@` operator to analyse numeric iterables into two fuzzy sensitive attributes;
the first of those contains a normalization to the range [0,1] and the
second its complement.

```python
import fairbench as fb
age = [18, 20, 19, 42, 30, 60, 18, 50, 40]
sensitive = fb.Dimensions(gender=fb.fuzzy @ age)
print(sensitive)
```
```text
genderlarge 60.000: [0.         0.04761905 0.02380952 0.57142857 0.28571429 1.
 0.         0.76190476 0.52380952]
gendersmall 18.000: [1.         0.95238095 0.97619048 0.42857143 0.71428571 0.
 1.         0.23809524 0.47619048]
```


## Intersectionality

When dealing with multiple sensitive attributes, 
dimensions for different attributes will often have 
overlapping non-zero values. This means that
certain groups may intersect. For example,
some blacks may also be women.
To consider intersectional definitions of fairness, 
create all dimension combinations with at least one 
data sample by using the intersectional method
of dimensions.

```python
import fairbench as fb
sensitive = fb.Dimensions(
    gender=fb.categories@["Man", "Woman", "Man", "Nonbin"],
    race=fb.categories@["Other", "Black", "White", "White"]
)
sensitive = sensitive.intersectional()
```

The method for creating intersectional groups accepts parameter *min_size*
that is the minimum number of data samples in each intersection.
By default, that has value of 1 to admit intersections of at least one
element, but you can set higher thresholds to prevent too small
subgroups from creating non-robust analys. 
You can also set it that value zero to obtain all group intersections.
If you are working with continuously-valued sensitive attributes,
this is the sum of scores across considered groups.

Note that intersectional analysis considers both the intersections
and the original groups, because there is a chance that biases
could be uncovered only when looking at broader group levels.
If you want to keep only a set of groups by removing all those
that have been split into, further apply `.strict()`. An example
of these concepts follows:


```python
import fairbench as fb
x, y, yhat = fb.bench.tabular.bank()
sensitive = fb.Dimensions(marital=fb.categories @ x["marital"], education=fb.categories @ x["education"])
sensitive = sensitive.intersectional(min_size=100).strict()
print(sensitive)  # only a few large subgroups are retained
```

```commandline
maritaldivorced&educationsecondary [0 0 0 ... 0 0 0]
maritalsingle&educationsecondary [0 1 0 ... 0 0 0]
maritalsingle&educationtertiary [0 0 0 ... 0 0 0]
maritalmarried&educationsecondary [1 0 0 ... 1 1 0]
maritalmarried&educationprimary [0 0 1 ... 0 0 1]
maritalmarried&educationtertiary [0 0 0 ... 0 0 0]
```

!!! warning
    Intersectional analysis already considers all subgroup
    combinations and not only pairwise ones. It therefore
    runs in time *O(m! n)* where *n* is the
    number of samples in the population and *m* the
    number of groups. Applying it twice in 
    succession has no value (no new groups would be
    found), and is **computationally intractable**
    if many interesections exist.
    
