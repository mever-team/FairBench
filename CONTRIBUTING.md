# Contributing
FairBench was designed to be easily extensible.
We are primarily looking to integrate new or existing
fairness metrics or reduction strategies, 
but improvements to the
codebase, suggestions, or github issues are welcome. :smile:

For a timely processing of a new pull request,
it is preferred (but not mandatory) to have a discussion
in the repository's issues first. You can also e-mail
the project's current maintainer: *Emmanouil (Manios) Krasanakis*
at *maniospas@hotmail.com* .

## Pull checklist

Follow these steps to add new features:

1. Fork the repository.
2. Clone the fork in your local development environment.
3. Install dependencies from the *.txt* lists. For example: `pip install -r requirements` installs the base library without the dependencies needed to run extras.
4. Write tests for new code, apply `black .` linting, and push the changes in your fork. 
5. Create a pull request from github's interface. You may want to mark that as draft.

## Codebase contributions

Contributions to *fairbench/bench* should make sure to not import modules unless code is explicitly called.
This lets the lightweight installation work without any extras. Base measures and comparison mechanisms can 
be found under *fairbench/v2/blocks*. Use existing implementations for reference, including decorators. Similarly,
implement filters under the *fairbench/v2/investigate* directory. 

Visualization environments reside under *fairbench/v2/export*, where a common data conversion mechanism is
used to synchronize messages between reports and the environments by traversing the former and calling 
methods of the latter. All visualization mechanisms should implement the same features to remain
compatible with each other. As an exception, serialization mechanisms are allowed to implement a `direct_show`
method that skips the common ground. Notice that this makes them lose several text formatting semantics - which
they would not be able to express anyway.

:bulb: *It is easiest to contribute with building blocks, as filters and visualization engines require walking through values with a good grasp on how these are generated and manipulated (see below).*

## :warning: Risky contributions

Contributions will be strictly reviewed when targeting the following directories,
as it can be very tricky to recover from failures found in respective code segments:

- Edits to the *fairbench/v1* directory will not be accepted unless they are made on features that have
not been completely phased out. Documentation does not cover this version of the interface anymore either.
- Editing the *fairbench/v2/core* directory will be accepted only upon exceptional circumstances, as
its code is heavily opinionated on how to both be dynamic and let *errors be comprehensive*.

## Working with fairbench values

You may want to leverage FairBench's inherent value
exploration and visualization model in your projects.In general, values consist of three parts: a value, a descriptor that contains
measurement units, roles, and descriptions, and dependent values. Then, they can be visualized with all the standard
mechanisms provided for reports. Here is a short example
of generating values:

```python
import fairbench as fb

descriptor = fb.core.Descriptor("MM", role="measure", details="my measure")
value = fb.core.Value(
  fb.core.TargetedNumber(0.5, target=1, bound=2, units="mm"), 
  descriptor=descriptor
)
print(value)
value.show(env=fb.export.Html) # shows the value and dependencies in the browser
```

```text
[measure] MM                             0.500 mm (ideal value 1.000, abs bound 2.000)
```


<details>
<summary>More on creating custom values.</summary>

The value itself can contain
more information (see below) or even not be provided at all (also see below). Let us check the simplest case:

```python
import fairbench as fb

value = fb.core.Value(0.5)
print(value)
```

```text
[any role] unknown                       0.500 
```

Add a descriptor for a demo measure "MM" (standing for 
"My Measure") like below. Of the descriptor's keyword arguments,
only the name and role are mandatory. The description appears
only in some of the more verbose visualizations, or for
descriptions obtained via the *help* function. 

```python
import fairbench as fb

descriptor = fb.core.Descriptor("MM", role="measure", details="my measure")
value = fb.core.Value(0.5, descriptor=descriptor)
print(value)
value.help()
```

```text
[measure] MM                             0.500 
##### FairBench help #####
This is my measure.
```

Add dependent values like below.

```python
import fairbench as fb

mm_descriptor = fb.core.Descriptor("MM", role="measure", details="my measure", preferred_units="mm")
first_half = fb.core.Value(0.25, descriptor=fb.core.Descriptor("first half", role="base"))
second_half = fb.core.Value(0.25, descriptor=fb.core.Descriptor("second half", role="base"))
value = fb.core.Value(0.5, descriptor=descriptor, depends=[first_half, second_half])
value.show()
```

```text
##### MM #####
|This is my measure.
|Value: 0.500 

  (0.0, 0.25)
  ▎ █  █
  ▎ █  █
  ▎ █  █
  ▎ █  █
  ▎ █  █
  ▎▬*▬▬-
  (2.0, 0.0)
  
   * first half                          0.250 
   - second half                         0.250 
```

Finally, it is important to add target values and units to numbers, to
ensure that results remain legible and comparable.
To do so, provide an `fb.core.Number` or `fb.core.TargetNumber` instead
of a float or no value. Custom numbers are shown below, where
the two accept the same arguments with the exception of *target* to
indicate the ideal value a measure should obtain. The *bound* the maximum
absolute value of the unit value (e.g., below bound 2 indicates that
the value lies in the range [-2,2]).

```python
import fairbench as fb

descriptor = fb.core.Descriptor("MM", role="measure", details="my measure")
value = fb.core.Value(fb.core.TargetedNumber(0.5, target=1, bound=2, units="mm"), descriptor=descriptor)
print(value)
```

```text
[measure] MM                             0.500 mm (ideal value 1.000, abs bound 2.000)
```

</details>