# :pencil: Contributing
`FairBench` was designed to be easily extensible.
We are primarily looking to integrate new or existing
fairness metrics, but any improvement to the
codebase is welcome. :smile:

For a timely processing of a new pull request,
it is preferred (but not mandatory) to have a discussion
in the repository's issues first. You can also e-mail
the project's current maintainer: Manios Krasanakis
at maniospas@hotmail.com .

## Pull checklist
1. Fork the repository.
2. Clone the fork in your local development environment.
3. Install dependencies.
4. Write tests for new code and push the changes in your fork. 
5. Create a pull request from github's interface.

## Create a new metric

1. Create it under `fairbench.metrics` module. 
2. Add the `parallel` decorator like this:
```
from faibench import parallel

@parallel
def metric(...):
    return ...
```
3. Reuse as many arguments found in other metrics as possible. 

:warning: Numeric inputs are automatically converted into 
[eagerpy](https://github.com/jonasrauber/eagerpy) tensors,
which use a functional interface to ensure interoperability.
You can use the `@parallel_primitive` decorator to avoid
this conversion. This lets you work with specific primitives 
provided by those working with your method, but try not to do so
without good reason, as it reduces computational equivalence
between environments.

:bulb: If your metric should behave differently for different 
data branches, add a `branch: str = None` default argument in its
definition to get the branch name.

## Create new reduction strategies
Reduction strategies follow three steps: transformation,
expansion, and reduction. To see how to orchestrate
these components, read [here](docs/reports.md) .


Expansion methods can be found
in the `fairbench.reports.reduction.expanders` module
and should transform a list into a (typically longer)
new list that stores the outcome of comparing the elements
of the original list, for instance pairwise. Start
by enriching the following expander definition:

```python
def expander(values: Iterable[ep.Tensor]) -> Iterable[ep.Tensor]:
    assert isinstance(values, list)
    return ... # a symmetric transformation of the first list here
```

Reducers take lists of values, such as lists produced by
and expander, and perform an aggregation strategy to summarize
it into one float value. Take care for your computations
to be backpropagateable. Add reducers in the 
`fairbench.reports.reduction.reducers` module and start
by enriching the following definition:

```python
def reducer(values: Iterable[ep.Tensor]) -> ep.Tensor:
    assert isinstance(values, list)
    return ... # float value that aggregates the list's elements
```
