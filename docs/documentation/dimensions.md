# Dimensions

`Dimensions` is a flexible data structures that stores 
multidimensional values, such as
sensitive attributes or predictions and labels for
multiple classes. Each dimension holds its own
vector of values. 

For example, you might have a 
dimension for each gender or race, or intersection thereof.
Each dimension vector holds fuzzy (in the range [0,1]) or 
binary values indicating the presence of the respective
attribute in data samples.

The most common use case is to
define a multidimensional sensitive attribute 
for intersectional fairness, like below. There,
the `fb.categories@` operator unpacks genders and
races from a discrete iterable into dictionaries
of binary-valued vectors and then these are combined
into one multidimensional object.

```python
import fairbench as fb

gender = ["Male", "Male", "Female", "Female", "Nonbinary"]  # any iterable
race = ["Black", "White", "White", "Black", "White"]  # any iterable
sensitive = fb.Dimensions(fb.categories@ gender, fb.categories@ race) 
sensitive = sensitive.intersectional() 
print(sensitive)
```

The outcome is the following. Note that, note that
the `intersectional` transformation
has enriched the original dimensions *White, Black, etc.* to also 
account for attribute value intersections, 
like *Male&White, Male&Black, etc.*

<pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">
White                          [0 1 1 0 1]
Black                          [1 0 0 1 0]
Male                           [1 1 0 0 0]
Male&White                     [0 1 0 0 0]
Male&Black                     [1 0 0 0 0]
Nonbinary                      [0 0 0 0 1]
Nonbinary&White                [0 0 0 0 1]
Female                         [0 0 1 1 0]
Female&White                   [0 0 1 0 0]
Female&Black                   [0 0 0 1 0]
</pre>


!!! info
    FairBench can unpack a wide range of data types as dimension values.
    These types include lists, arrays, and tensors of popular machine learning frameworks.


## Explicit dimensions
For values that are non-dictionaries,
keywords like `men`, `women`, and `nonbinary` are dimension names. 
Dimension values will usually be lists,
numpy arrays or deep learning tensors. 
Provide any number of dimensions
with any names. 

```python
import fairbench as fb

sensitive = fb.Dimensions(
    men=[1, 1, 0, 0, 0], 
    women=[0, 0, 1, 1, 0],
    nonbinary=[0, 0, 0, 0, 1]
)
```

Access dimension vectors 
as either members of the `Dimensions` object
or as dictionary entries. Those vectors are
stored as [eagery](https://github.com/jonasrauber/eagerpy) 
tensors. Use eagerpy methods on `Dimensions` 
to obtain a transformation with different values for each
dimension. Mainly, this is useful for summing attribute
vector values to get a sense of corresponding demographic
group sizes.

```
print(sensitive.nonbinary.numpy())
print(sensitive["nonbinary"].numpy())) # does the same
print(sensitive.sum())
```

This yields the following output:

<pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">
NumPyTensor(array([0, 0, 0, 0, 1]))
NumPyTensor(array([0, 0, 0, 0, 1]))
men                            2
women                          2
nonbinary                      1
</pre>

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
print(sensitive.women.numpy()) # .numpy() is an eagerpy method
print(sensitive["non-binary"].numpy())
```

<pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">
[0, 0, 1, 1, 0]
[0, 0, 0, 0, 1]
</pre>

Freely mix dimensions of multiple sensitive attributes.
That is, treat each value of each dimension as a separate dimension.
For example, mix in gender and age sensitive attributes
in the same `Dimensions` object like so.

```python
import fairbench as fb

sensitive = fb.Dimensions(
    men=[1, 1, 0, 0, 0],
    nonmen=[0, 0, 1, 1, 1],
    IsOld=[0, 1, 0, 1, 0]
)
```

To add multiple sensitive attributes without worrying about 
conflicting names, pass dictionaries as keyword 
arguments. This prepends the keyword argument name to 
all generated branch names.

```python
import fairbench as fb

sensitive = fb.Dimensions(
    gender={"1": [0, 0, 1, 1, 0], "0": [1, 1, 0, 0, 0], "?": [0, 0, 0, 0, 1]},
    isold={"1": [0, 1, 0, 1, 0], "0": [1, 0, 1, 0, 1]}
)
```

This yields the following output:

<pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">
gender1                        [0 0 1 1 0]
gender0                        [1 1 0 0 0]
gender?                        [0 0 0 0 1]
isold1                         [0 1 0 1 0]
isold0                         [1 0 1 0 1]
</pre>

## Unpack to dictionaries

FairBench offers helper operators toconvert iterable
data into dictionaries that can be passed to `Dimensions`.

The most common pattern is analyzing categorical 
values found in iterables with the
`fb.categories@` operator.
For example, when applied on a list whose entries are among
"Man". "Woman", "Nonbin"
this creates three dimensions storing binary membership for each
of those genders.

```python
import fairbench as fb

sensitive = fb.Dimensions(fb.categories@["Man", "Woman", "Man", "Woman", "Nonbin"])
print(sensitive)
```

This yields the following:

<body><pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">
Woman                          [0 1 0 1 0]
Man                            [1 0 1 0 0]
Nonbin                         [0 0 0 0 1]
</pre>

Add the outcomes of multiple category analyses 
to `Dimensions` with the patterns
already seen.

```python
import fairbench as fb

gender = ...  # iterable (e.g., list) of gender attribute for each data sample
race = ...  # iterable (e.g., list) of race attribute for each data sample
sensitive = fb.Dimensions(
    gender=fb.categories@ gender, 
    race=fb.categories@ race
) 
```

!!! info
    Any Python iterable can be analyzed into categories.
    This includes lists, pandas dataframe
    columns, categorical tensors, and numpy arrays.

Use the `fuzzy@` operator to unpack numeric iterables into two 
fuzzy sensitive attributes; the first of those contains a 
normalization to the range [0,1] and the second its complement.

```python
import fairbench as fb
age = [18, 20, 19, 42, 30, 60, 18, 50, 40] # in years
sensitive = fb.Dimensions(age=fb.fuzzy@ age)
print(sensitive)
```

This prints the following. This particular fuzzy expansion creates
two complementary numbers.

<pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">
agelarge 60.000                [0.         0.04761905 0.02380952 0.57142857 0.28571429 1.
 0.         0.76190476 0.52380952]
agesmall 18.000                [1.         0.95238095 0.97619048 0.42857143 0.71428571 0.
 1.         0.23809524 0.47619048]
</pre>

!!! info
    All FairBench measures accept fuzzy-based weighting
    of group membership by treating fuzzy numbers as the probability
    of exhibiting value of 1 and outputting the average expected
    value of computations.


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

<pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">
maritalsingle&educationtertiary [0 0 1 ... 0 0 0]
maritalsingle&educationprimary [0 0 0 ... 0 0 0]
maritalsingle&educationsecondary [0 0 0 ... 0 0 0]
maritalsingle&educationunknown [0 0 0 ... 0 0 0]
maritalmarried&educationtertiary [0 0 0 ... 0 0 0]
maritalmarried&educationprimary [0 0 0 ... 1 0 1]
maritalmarried&educationsecondary [0 1 0 ... 0 1 0]
maritalmarried&educationunknown [0 0 0 ... 0 0 0]
maritaldivorced&educationtertiary [0 0 0 ... 0 0 0]
maritaldivorced&educationprimary [0 0 0 ... 0 0 0]
maritaldivorced&educationsecondary [1 0 0 ... 0 0 0]
</pre>

!!! danger
    Intersectional analysis already considers all subgroup
    combinations and not only pairwise ones. It therefore
    runs in time *O(m! n)* where *n* is the
    number of samples in the population and *m* the
    number of groups. Applying it twice in 
    succession has no value (no new groups would be
    found), and is **computationally intractable**
    if many interesections exist.
    
