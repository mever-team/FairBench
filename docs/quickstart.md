# Quickstart

Before starting, install FairBench with:

```shell
pip install --upgrade fairbench
```

A typical workflow using the library will help you identify prospective
fairness concerns to discuss with stakeholders. Decide which of 
the concerns matter after drawing a broad enough
picture. Follow the steps below.

## 1. Prepare data

Run your system to generate some predictions for test data.
Here, we assess biases for a demo binary classifier and dataset,
but other types of predictions can be analysed too. 
Supported data formats include lists, numpy arrays, 
and pytorch/tensorflow/jax tensors.

```python
import fairbench as fb
test, y, yhat = fb.bench.tabular.adult()
```

## 2. Sensitive attributes

Pack sensitive attributes found in your test data
into a data structure holding multiple [dimensions](documentation/dimensions.md).
This stores any number of attributes with any number of values
by considering each value as a separate dimension.
One construction pattern is the following:

```python
sensitive = fb.Dimensions(fb.categories @ test[8], fb.categories @ test[9])  # analyse the gender and race columns
sensitive = sensitive.intersectional()  # automatically find non-empty intersections
sensitive = sensitive.strict()  # keep only intersections that have no children
```

## 3. Compute reports

Use sensitive attribute forks alongside predictions 
to generate fairness reports.
Below is a pairwise report, which compares all pairs
of population groups or subgroups defined in the sensitive attribute
based on a wide range of base performance measures. 
Reports can be viewed under various visualization environments.

The comparisons for each measure are reduced to one value
with various reduction strategies (the columns).
The task type (here: binary classification)
and corresponding base performance metrics are determined
by the report's arguments.


```python
report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
report.show(env=fb.export.ConsoleTable)  
```

```text
                                    multidim                                                                   
                                           acc           pr          tpr          tnr          tar          trr
min                                      0.876        0.056        0.348        0.966        0.036        0.822
max                                                   0.097                                  0.067        0.884
maxerror                                 0.124                     0.652        0.034                          
wmean                                    0.906        0.066        0.382        0.975        0.044        0.862
mean                                     0.900        0.077        0.407        0.972        0.052        0.847
maxrel                                   0.048        0.422        0.304        0.012        0.465        0.070
maxdiff                                  0.044        0.041        0.152        0.012        0.031        0.062
gini                                     0.011        0.118        0.083        0.003        0.133        0.016
std                                      0.018        0.017        0.067        0.005        0.013        0.027
```


## 4. Go into details

Explore reports by focusing on any of their contributing
computations with the dot notation programmatically,
or with interactive visualization environments.
You may also add more `depth` to
their view. Below is an example, but there are
many dynamic options [here](documentation/interactive.md).
We focus on only the minimum accuracy to keep the outcome simple,
but visualization environments
work with complicated reports too.

```python
report.min.acc.show(env=fb.export.Console)
```

```text
##### min acc #####
|This reduction of a measure is the minimum of the accuracy.
|Value: 0.866 min acc

  (0.0, 0.9213973799126638)
  ▎       █
  ▎ ▄  ▆  █
  ▎ █  █  █
  ▎ █  █  █
  ▎ █  █  █
  ▎▬*▬▬-▬▬+
  (3.0, 0.0)
  
   * single                              0.866 acc
   - married                             0.901 acc
   + divorced                            0.921 acc
```

## 5. Simplify reports 

Apply filters to focus on specific types of evaluation,
like keeping computations that show only bias
or keeping only bias/fairness values violating
certain thresholds.

One of the available filters, which is presented
below, are fairness stamps. These refer to a few 
common types of fairness evaluation and are accompanied
by caveats and recommendations. The collection of available
stamps is called a fairness modelcard, though it is
a normal report and can be manipulated (e.g., viewed) 
normally.

```python
report.filter(fb.investigate.Stamps).show(env=fb.export.Html, depth=1)
```

*The output can be viewed [here](documentation/example_html.html).*

The `Html` environment that was used this time
can save and/or open in the browser
the generated HTML representation of reports. The generated document
requires an internet connection to properly view, as it depends
on [bootstrap](https://getbootstrap.com/) for theming. 
It is equivalent to the `Console` environment of previous examples.
The provided depth controls the level of details (default is zero).

!!! danger
    Blindly applying filters may neglect certain
    kinds of evaluation.

