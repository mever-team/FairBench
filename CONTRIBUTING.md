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
3. Install dependencies.
4. Write tests for new code and push the changes in your fork. 
5. Create a pull request from github's interface.

## Codebase contributions

Contributions to *fairbench/bench* should make sure to not import modules unless code is explicitly called.
This lets the lightweight installation work without any extras. Base measures and comparison mechanisms can 
be found under *fairbench/v2/blocks. Use existing implementations for reference, including decorators. Similarly,
implement filters under the *fairbench/v2/investigate* directory.

Visualization environments reside under *fairbench/v2/export*, where a common data conversion mechanism is
used to synchronize messages between reports and the environments by traversing the former and calling 
methods of the latter. All visualization mechanisms should implement the same features to remain
compatible with each other. As an exception, serialization mechanisms are allowed to implement a `direct_show`
method that skips the common ground. Notice that this makes them lose several text formatting semantics - which
they would not be able to express anyway.

Contributions will be strictly reviewed when targeting the following directories,
as it can be very tricky to recover from failures found in respective code segments:

- Editing the *fairbench/v1* directory will not be accepted unless they are made on features that have
not been completely phased out. Documentation does not cover this version of the interface anymore either.
- Editing the *fairbench/v2/core* directory will be accepted only upon exceptional circumstances, as it
its code is heavily opinionated and follows a complex flow to make sure that *errors are always comprehensible*.
