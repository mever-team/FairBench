# Forks

Forks are flexible data structures that, among other things, store
and manage multidimentional sensitive attributes. 
They are comprised of named data, called branches. 
For instance, there
may a branch for each gender or race. These would be
binary arrays of the same length that capture which 
data samples exhibit protected attribute values.


## Creating forks

<button onclick="toggleCode('code1')" class="toggle-button">>></button>
Generate forks by passing keyword
arguments to a constructor. 
In the snippet below `men`, `women`, and `nonbinary` are
branch names. Branch values can be anything,
though they will usually be arrays.
Provide any number of branches
with any names and access their values 
like object members.
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
To set branch names programmatically
or to strings with invalid variable characters
pass a dictionary that maps names to values 
as a positional argument. You can do this in addition
to branches declared via keyword arguments.
<div id="code2" class="code-block" style="display:none;">
```python
import fairbench as fb
import numpy as np
sensitive = fb.Fork({"non-binary": np.array([0, 0, 0, 0, 1])}, 
                    men=np.array([1, 1, 0, 0, 0]), 
                    women=np.array([0, 0, 1, 1, 0]))
```
</div>


<button onclick="toggleCode('code3')" class="toggle-button">>></button>
For more than one sensitive
attribute, add the branches you would declare for
every attribute more branches of the same fork.
For instance, a fork can considers any number of
gender attribute values and a binary sensitive attribute 
value indicating young vs old people.
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
Pass dictionaries as positional arguments to
prepend the argument name to all the branch names. Use this to
add multiple sensitive attributes at once without worrying
about conflicting branch names.
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

To create forks by analysing categorical values found in iterables 
prepend the latter with the `categories@` operator.
For example, the following code snippet
creates two branches `genderMan,genderWoman` and stores binary
membership to each of those. 

```python
fork = fb.Fork(gender=fb.categories@["Man", "Woman", "Man", "Woman", "Nonbin"])
print(fork)
# genderMan: [1, 0, 1, 0, 0]
# genderWoman: [0, 1, 0, 1, 0]
# genderNonbin: [0, 0, 0, 0, 1]
```

Add the outcomes of any number of 
category analyses to a fork. Use
positional arguments
(instead of named keyword arguments, such as `gender` in the above example)
to avoid prepending the keyword argument's name to branch names.
Any iterable can be analysed into categories instead of a list, 
including categorical tensors or arrays.


## Intersectionality

For more than one sensitive attributes,
branches capturing the values of different attributes
will have overlapping non-zeroes.
Thus, you might want to consider intersectional 
instead of more naive definitions of fairness
by creating all branch combinations with at least
one data sample per:

```python 
sensitive = sensitive.intersectional()
```




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