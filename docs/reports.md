# Full reports

Before starting, install FairBench with:

```shell
pip install --upgrade fairbench
```

This is only the lightweight version without any specifications,
which suffices for assessment of any system - install extras are
needed if you want run out-of-the-box benchmarks for vision data or LLMs.
A typical workflow using the library will help you identify prospective
fairness concerns to discuss with stakeholders. Decide which of 
the concerns matter after drawing a broad enough
picture. Follow the steps below.

## 1. Prepare data

To assess your system, use it to generate predictions for test data.
Here, we look at a binary classifier and dataset,
but other types of predictions can be assessed for bias too. 
Supported data formats include lists, numpy arrays, 
and pytorch/tensorflow/jax tensors.

```python
import fairbench as fb
test, y, yhat = fb.bench.tabular.adult()
```

## 2. Sensitive attributes

Pack sensitive attributes found across test samples
into a data structure holding multiple [dimensions](documentation/dimensions.md).
This stores any number of attributes with any number of values
by considering each value as a separate dimension.

In particular, each dimension is represented as a binary or fuzzy array
whose i-th element represents whether the i sample has the attribute
corresponding to the dimension. 
One construction pattern is the following. This first analyses categorical
iterables and then packs them into dimensions. It then computes all non-empty
intersections that combine attributes, and finally retains the most specialized 
intersectional subgroups.

```python
sensitive = fb.Dimensions(fb.categories @ test[8], fb.categories @ test[9])  # analyse the gender and race columns
sensitive = sensitive.intersectional()  # automatically find non-empty intersections
sensitive = sensitive.strict()  # keep only intersections that have no children
```

## 3. Compute reports

Use sensitive attributes alongside predictions 
to generate fairness reports.
Below is one that compares all pairs
of population groups or subgroups 
according to a wide range of base performance measures. 
Reports can be viewed under various visualization environments.

The comparisons for each measure (row) are reduced to one value
with reduction strategies (columns).
The task type and applicable measures are determined
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
computations. Use the programmatic dot notation,
where more `depth` can be added to viewed values
to further expand intermediate computations and search
for the root causes of discrimination. 
Below is an example, but there are also
many dynamic visualization options [here](documentation/interactive.md).

In the example, we focus on only the minimum accuracy to keep the outcome simple,
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

FairBench provides filters that enhance the evaluation.
For example, they can remove report entries that do not violate certain
threshold, leaving only problematic values behind.

One of the available filters, which is presented
below, are fairness stamps. These refer to a few 
common types of fairness evaluation and are accompanied
by caveats and recommendations. The collection of available
stamps is called a fairness modelcard. Filtering through
reports yields new ones that can be manipulated (e.g., viewed) 
normally.

The `Html` environment that is used below
can save and/or open in the browser
the generated HTML representation of reports. The generated document
requires an internet connection to properly view, as it depends
on [bootstrap](https://getbootstrap.com/) for theming. 
It is equivalent to the `Console` environment of previous examples.
The provided depth controls the level of details (default is zero).

```python
report = report.filter(fb.investigate.Stamps)
report.show(env=fb.export.Html(horizontal=True), depth=1)
```

![stamps](stamps.png)


!!! danger
    Blindly following filters may neglect certain
    kinds of evaluation. Icons also give an indication
    of which are the most problematic values, 
    but any measure value could be worrisome.
    Therefore, computed values are also shown.

