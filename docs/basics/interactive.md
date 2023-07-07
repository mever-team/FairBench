# Interactive visualization

Interactive visualization explores
complex objects generated with  `FairBench`, such as
forks of forks of measure values (these can compare
the same reports across multiple algorithms).

1. [Report perspectives](#report-perspectives)
2. [Start visualization](#start-visualization)
3. [Interface](#interface)

## Report perspectives

Before learning about interactive visualization
that can create complicate comparisons between
subgroups, it is important to understand how
different report perspectives can be obtained by
taking advantage FairBench's 
[concurrent execution](../advanced/distributed.md#computational-branches).

Mathematically, extracting report persoectives
is equivalent to tensor rearrangement and element acces,
though with easier-to-follow dimension names.

!!! info 
    This paragraph is under construction.

## Start visualization

To start interactive visualization, call 
`fairbench.interactive(obj)` on an object
`obj` that is a dictionary or fork.

From a console, this will start a bokeh server
that hosts a dynamic web page and will open this
as a tab in your local browser (hard-terminate
the process to stop the server). From a 
Jupyter environment, a bokeh application will
run on the next output cell instead.

## Interface

!!! info 
    This paragraph is under construction.
