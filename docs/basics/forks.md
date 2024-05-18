# Forks

Forks are flexible data structures that store 
and manage multidimensional sensitive attributes. 
They are made up of named data elements called branches. 
Each branch represents a specific sensitive attribute 
value. For example, you might have a 
branch for each gender, which would be a binary 
array indicating the presence of that attribute 
in data samples.

Click on the buttons below to see how to create
forks with various coding patterns.
As a preview of a common case, here's how to 
define a sensitive attribute fork for intersectional 
fairness (for a full example using this definition
look at the [quickstart](../quickstart.md)):

```python
gender = ...  # iterable (e.g., list) of gender attribute for each data sample
race = ...  # iterable (e.g., list) of race attribute for each data sample
sensitive = fb.Fork(fb.categories @ gender, fb.categories @ race) 
sensitive = sensitive.intersectional() 
```


## Creating forks

<button onclick="toggleCode('code1')" class="toggle-button">>></button>
Generate forks by passing keyword
arguments to a constructor. 
For example,  `men`, `women`, and `nonbinary` are
branch names. Branch values can be anything,
though they will usually be lists,
numpy arrays or deep learning tensors. 
Provide any number of branches
with any names and access their values 
like members of the fork object.
<div id="code1" class="code-block" style="display:none;">
```python
import fairbench as fb
import numpy as np
sensitive = fb.Fork(men=np.array([1, 1, 0, 0, 0]), 
                    women=np.array([0, 0, 1, 1, 0]),
                    nonbinary=np.array([0, 0, 0, 0, 1]))
print(sensitive.nonbinary)
#    [0, 0, 0, 0, 1]
```
</div>

<button onclick="toggleCode('code2')" class="toggle-button">>></button>
To set branch names programmatically or use names 
with invalid characters, pass a dictionary mapping 
names to values as a positional argument. 
You can do this in addition to branches declared 
via keyword arguments. Access branch names as strings
by treating the fork as a dictionary.

<div id="code2" class="code-block" style="display:none;">
```python
import fairbench as fb
import numpy as np
sensitive = fb.Fork({"non-binary": np.array([0, 0, 0, 0, 1])}, 
                    men=np.array([1, 1, 0, 0, 0]), 
                    women=np.array([0, 0, 1, 1, 0]))
print(sensitive.men)  # [1, 1, 0, 0, 0]
print(sensitive["men"])  # the same as above
```
</div>


<button onclick="toggleCode('code3')" class="toggle-button">>></button>
For multiple sensitive attributes and attribute values, 
add branches for each attribute value to the same fork. 
For instance, you can have branches for different 
gender values and a binary attribute for age.
<div id="code3" class="code-block" style="display:none;">
```python
import fairbench as fb
import numpy as np
sensitive = fb.Fork(men=np.array([1, 1, 0, 0, 0]),
                    nonmen=np.array([0, 0, 1, 1, 1]), , 
                    IsOld=np.array([0, 1, 0, 1, 0]))
```
</div>


<button onclick="toggleCode('code4')" class="toggle-button">>></button>
To add multiple sensitive attributes without worrying about 
conflicting branch names, pass dictionaries as positional 
arguments. This prepends the argument name to 
all generated branch names.
<div id="code4" class="code-block" style="display:none;">
```python
import fairbench as fb
import numpy as np
sensitive = fb.Fork(gender={"1": np.array([0, 0, 1, 1, 0]),
                            "0": np.array([1, 1, 0, 0, 0]),
                            "?": np.array([0, 0, 0, 0, 1])},
                    isold={"1": np.array([0, 1, 0, 1, 0]),
                           "0": np.array([1, 0, 1, 0, 1])})
```
</div>


!!! info
    When working with specific 
    [backends](../advanced/ml_integration.md#backend-selection),
    branch values are internally converted to appropriate data types
    (e.g., arrays or tensors).  

## Unpacking

<button onclick="toggleCode('code5')" class="toggle-button">>></button>
To create forks by analyzing categorical 
values found in iterables, use the 
`categories@` operator.
For example, appled on a list with entries "Man" and "Woman"
this operator creates two branches storing 
binary membership for each.

<div id="code5" class="code-block" style="display:none;">
```python
import fairbench as fb
fork = fb.Fork(
 fb.categories@["Man", "Woman", "Man", "Woman", "Nonbin"])
print(fork)
# genderMan: [1, 0, 1, 0, 0]
# genderWoman: [0, 1, 0, 1, 0]
# genderNonbin: [0, 0, 0, 0, 1]
```
</div>

<button onclick="toggleCode('code6')" class="toggle-button">>></button>
You can add the outcomes of multiple category analyses 
to a fork. Use named keyword arguments to prepend the 
that name to branch names, or put all category analyses
as positional arguments to just merge their branches.
Any Python iterable can be analyzed into categories.
This includes lists, pandas datafragme
columns, categorical tensors, or numpy arrays.

<div id="code6" class="code-block" style="display:none;">
```python
gender = ...  # iterable (e.g., list) of gender attribute for each data sample
race = ...  # iterable (e.g., list) of race attribute for each data sample
sensitive = fb.Fork(gender=fb.categories @ gender, 
                    race=fb.categories @ race) 
```
</div>

## Intersectionality

When dealing with multiple sensitive attributes, 
branches for different attributes will often have 
overlapping non-zero values. This means that
certain groups may intersect. For example,
some blacks may also be women.


<button onclick="toggleCode('code7')" class="toggle-button">>></button>
To consider intersectional definitions of fairness, 
create all branch combinations with at least one 
data sample by using the intersectional method
of forks.

<div id="code7" class="code-block" style="display:none;">
```python
import fairbench as fb
sensitive = fb.Fork(gender=fb.categories@["Man", "Woman", "Man", "Nonbin"],
                    race=fb.categories@["Other", "Black", "White", "White"])
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
import fairbench as fbimport fairbench as fb
sensitive = fb.Fork(fb.categories@["Man", "Woman", "Man", "Nonbin"]
                    & fb.categories@["Black", "Black", "White", "White"])
print(sensitive)
```

```
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