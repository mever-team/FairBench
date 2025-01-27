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


## Create dimensions

There are several patterns for constructing dimensions.

<button onclick="toggleCode('code1')" class="toggle-button">>></button>
<b>Keyword arguments</b><br> For values that are non-dictionaries,
keywords like `men`, `women`, and `nonbinary` are dimension names. 
Dimension values will usually be lists,
numpy arrays or deep learning tensors. 
Provide any number of dimensions
with any names and access their values 
as either members of the `Dimensions` object
or dictionary values.
<div id="code1" class="code-block" style="display:none;">

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
</div>

<button onclick="toggleCode('code2')" class="toggle-button">>></button>
<b>From dictionaries</b><br>
To set dimensions names programmatically or use names 
with special characters, such as spaces, pass as a positional argument
to the constructor a dictionary 
that maps names to values. 
Do this with any number of dictionaries, and in addition to branches 
declared via keyword arguments.

<div id="code2" class="code-block" style="display:none;">

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
</div>


<button onclick="toggleCode('code3')" class="toggle-button">>></button>
<b>Multidimensional</b><br>
It is acceptable to have dimensions with different nature, 
such as multiple sensitive attributes and attribute values.
For example, mix in gender and age sensitive attributes
in the same `Dimensions` object.

<div id="code3" class="code-block" style="display:none;">

```python
import fairbench as fb

sensitive = fb.Dimensions(
    men=[1, 1, 0, 0, 0],
    nonmen=[0, 0, 1, 1, 1],
    IsOld=[0, 1, 0, 1, 0]
)
```
</div>


<button onclick="toggleCode('code4')" class="toggle-button">>></button>
<b>Disambiguate conflicting dimension names</b><br>
To add multiple sensitive attributes without worrying about 
conflicting branch names, pass dictionaries as positional 
arguments. This prepends the argument name to 
all generated branch names.
<div id="code4" class="code-block" style="display:none;">

```python
import fairbench as fb

sensitive = fb.Dimensions(
    gender={"1": [0, 0, 1, 1, 0], "0": [1, 1, 0, 0, 0], "?": [0, 0, 0, 0, 1]},
    isold={"1": [0, 1, 0, 1, 0], "0": [1, 0, 1, 0, 1]}
)
```
</div>


<button onclick="toggleCode('code6')" class="toggle-button">>></button>
<b>From multiple iterables</b><br>
Add the outcomes of multiple category analyses 
to `Dimensions`; use named keyword arguments to prepend the 
that name to dimensions names, or put all category analyses
as positional arguments to just merge their branches.
Any Python iterable can be analyzed into categories.
This includes lists, pandas dataframe
columns, categorical tensors, or numpy arrays.

<div id="code6" class="code-block" style="display:none;">

```python
import fairbench as fb

gender = ...  # iterable (e.g., list) of gender attribute for each data sample
race = ...  # iterable (e.g., list) of race attribute for each data sample
sensitive = fb.Dimensions(
    gender=fb.categories @ gender, 
    race=fb.categories @ race
) 
```
</div>


## Unpack dictionaries

Here are helper operators that can convert iterable
data into dictionaries to pass into `Dimensions`.

<button onclick="toggleCode('code5')" class="toggle-button">>></button>
<b>Categorical iterables</b><br>
Create multiple dimensions by analyzing categorical 
values found in iterables with the
`categories@` operator.
For example, when applied on a list whose entries are among
"Man". "Woman", "Nonbin"
this operator creates three branches storing binary membership for each
of those declared genders.

<div id="code5" class="code-block" style="display:none;">

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
</div>

<button onclick="toggleCode('code9')" class="toggle-button">>></button>
<b>Fuzzy continuous valued iterables</b><br>
Use the `fuzzy@` operator to analyse numeric iterables into two fuzzy sensitive attributes;
the first of those contains a normalization to the range [0,1] and the
second its complement.

<div id="code9" class="code-block" style="display:none;">

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
</div>



## Intersectionality

When dealing with multiple sensitive attributes, 
dimensions for different attributes will often have 
overlapping non-zero values. This means that
certain groups may intersect. For example,
some blacks may also be women.


<button onclick="toggleCode('code7')" class="toggle-button">>></button>
To consider intersectional definitions of fairness, 
create all dimension combinations with at least one 
data sample by using the intersectional method
of dimensions.

<div id="code7" class="code-block" style="display:none;">
```python
import fairbench as fb
sensitive = fb.Dimensions(
    gender=fb.categories@["Man", "Woman", "Man", "Nonbin"],
    race=fb.categories@["Other", "Black", "White", "White"]
)
sensitive = sensitive.intersectional()
```
</div>


<button onclick="toggleCode('code8')" class="toggle-button">>></button>
You may want to allow empty intersections, because
some report types can handle them. To do so, you should not
use the intersectional method but
explicitly combine the outcome of categorical analysis
for multiple attributes with the bitwise *and* `&`.

<div id="code8" class="code-block" style="display:none;">
```python
import fairbench as fb
sensitive = fb.Dimensions(
    fb.categories@["Man", "Woman", "Man", "Nonbin"]
    & fb.categories@["Black", "Black", "White", "White"]
)
print(sensitive)
```

```text
Woman&Black: [0 1 0 0]
Woman&White: [0 0 0 0]
Man&Black: [1 0 0 0]
Man&White: [0 0 1 0]
Nonbin&Black: [0 0 0 0]
Nonbin&White: [0 0 0 1]
```

</div>



<script>
function toggleCode(id) {
    var codeBlock = document.getElementById(id);
    if (codeBlock.style.display === "none") {
        codeBlock.style.display = "block";
    } else {
        codeBlock.style.display = "none";
    }
}
</script>