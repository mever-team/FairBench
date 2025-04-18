# Visualization

Reports accept visualization environments as arguments to their `show` method.
These environments can be either static or interactive, where the last
category requires FairBench to be installed with the `interactive` extra
dependencies:

```bash
pip install --upgrade fairbench[interactive]
```

## Console

The default visualization environment prints all report
details in the console (it also uses ansi color codes
that do not appear in this page). It can also be
selected by passing its class as an argument
to the `show` method, in which case it is instantiated
with default arguments.

```python
report.show(env=fb.export.Console)
```

The above makes use of the [ansiplot](https://github.com/maniospas/ansiplot)
library to have some nice plots. You can switch
to a less verbose style by manually instantiating
the environment, like below.

```python
report.show(env=fb.export.Console(ansiplot=False))
```

<div style="overflow-y: scroll;height: 380px; margin-bottom: 30px;">

```
##### multidim #####
|This is analysis that compares several groups.
|Computations cover several cases.

 ***** min *****
 |This reduction is the minimum.
 |Computations cover several cases.
   |min acc                            0.333      ███ 
   |min pr                             0.333      ███ 
   |min tpr                            0.000       
   |min tnr                            0.500      █████ 
   |min tar                            0.000       
   |min trr                            0.333      ███ 
 
 ***** max *****
 |This reduction is the maximum.
 |Computations cover several cases.
   |max pr                             0.500      █████ 
   |max tar                            0.500      █████ 
   |max trr                            0.500      █████ 
 
 ***** maxerror *****
 |This reduction is the maximum deviation from the ideal value.
 |Computations cover several cases.
   |maxerror acc                       0.667      ██████▌ 
   |maxerror tpr                       1.000      ██████████ 
   |maxerror tnr                       0.500      █████ 
 
 ***** wmean *****
 |This reduction is the weighted average.
 |Computations cover several cases.
   |wmean acc                          0.600      ██████ 
   |wmean pr                           0.400      ████ 
   |wmean tpr                          0.400      ████ 
   |wmean tnr                          0.700      ███████ 
   |wmean tar                          0.200      ██ 
   |wmean trr                          0.400      ████ 
 
 ***** mean *****
 |This reduction is the average.
 |Computations cover several cases.
   |mean acc                           0.667      ██████▌ 
   |mean pr                            0.417      ████ 
   |mean tpr                           0.500      █████ 
   |mean tnr                           0.750      ███████▌ 
   |mean tar                           0.250      ██▌ 
   |mean trr                           0.417      ████ 
 
 ***** maxrel *****
 |This reduction is the maximum relative difference.
 |Computations cover several cases.
   |maxrel acc                         0.667      ██████▌ 
   |maxrel pr                          0.333      ███ 
   |maxrel tpr                         0.000       
   |maxrel tnr                         0.500      █████ 
   |maxrel tar                         0.000       
   |maxrel trr                         0.333      ███ 
 
 ***** maxdiff *****
 |This reduction is the maximum difference.
 |Computations cover several cases.
   |maxdiff acc                        0.667      ██████▌ 
   |maxdiff pr                         0.167      █▌ 
   |maxdiff tpr                        1.000      ██████████ 
   |maxdiff tnr                        0.500      █████ 
   |maxdiff tar                        0.500      █████ 
   |maxdiff trr                        0.167      █▌ 
 
 ***** gini *****
 |This reduction is the gini coefficient.
 |Computations cover several cases.
   |gini acc                           0.250      ██▌ 
   |gini pr                            0.100      █ 
   |gini tpr                           0.500      █████ 
   |gini tnr                           0.167      █▌ 
   |gini tar                           0.500      █████ 
   |gini trr                           0.100      █ 
 
 ***** std *****
 |This reduction is the standard deviation.
 |Computations cover several cases.
   |std acc                            0.333      ███ 
   |std pr                             0.083      ▌ 
   |std tpr                            0.500      █████ 
   |std tnr                            0.250      ██▌ 
   |std tar                            0.250      ██▌ 
   |std trr                            0.083      ▌ 
```
</div>

## ConsoleTable

A more concise visualization strategy is to create
tables in the console, like below. This also admits
a `sideways` argument with default value *True*
that determines whether to show
multiple tables side-by-side if necessary.
It also accepts a `legend` argument with default False
about showing additional textual descriptions,
similarly to the console report.

```python
report.show(fb.export.ConsoleTable)
```

```
                                    multidim                                                                   
                                           acc           pr          tpr          tnr          tar          trr
min                                      0.333        0.333            0        0.500            0        0.333
max                                                   0.500                                  0.500        0.500
maxerror                                 0.667                         1        0.500                          
wmean                                    0.600        0.400        0.400        0.700        0.200        0.400
mean                                     0.667        0.417        0.500        0.750        0.250        0.417
maxrel                                   0.667        0.333            0        0.500            0        0.333
maxdiff                                  0.667        0.167            1        0.500        0.500        0.167
gini                                     0.250        0.100        0.500        0.167        0.500        0.100
std                                      0.333        0.083        0.500        0.250        0.250        0.083
```

!!! tip
    The default options of ConsoleTable provides the most concise representation of results.
    However, you might need refreshers about each entry, for example by adding a legend
    or running `report.help()`.

## Html

This is an equivalent to the Console environment that converts
presented text and quantities to a static HTML page. That page 
displays evaluation cards one under the other, or next to each
other as demonstrated in the [quickstart](../reports). 
Note that increasing the of the *show* method, for example to
`depth=2`, adds a lof of useful information but may take some
time to go through.

When instantiating the environment with non-default values,
use the `filename` argument to set a file path for exporting results. 
If this is *None*, the generated HTML text is returned from the
*show* method instead. Pass `view=False` if you want to write to the file
without showing anything. An example follows.

```python
html_text = report.show(fb.export.Html(filename=None, horizontal_bars=False)) 
```

If you have a wide enough screen, it may be more convenient to display 
the individual cards side-by-side by setting `horizontal=False` 
in the environment's constructor. An example is presented below.
By default, distributions are hidden under expanding 
details, but set `distributions=True` to make make them
always visible. Finally, set `horizontal_bars=False` to 
create bar plots with vertical bars; otherwise, horizontal
bars are used to account for scenarios where many values
are plotted to be compared. These options are demonstrated below.

```python
report.show(env=fb.export.Html(distributions=True, horizontal=True, horizontal_bars=True))
```

![html_horizontal.png](html_horizontal.png)


If you want only to only see the markings (checkmark, questionmark, or X) 
that appear for higher 
depths like the above, pass a `legend=False` argument to this environment. 
This is more verbose that HTMLTable. Here is an example.
THe markings indicate whether values are proximate to ideal
ones by an error of 0.25 in the default case - though certain 
[filters](filters.md) control the threshold. Therefore, checkmarks
do not necessarily indicate fairness, but just smaller magnitudes of
bias. However, those small biases may also be the ones most prohibitive 
in the deployment context of the evaluated AI systems.

```python
report.show(fb.export.Html(legend=False), depth=2)
```

![legend_false.png](legend_false.png)


## HtmlTable

This is an equivalent to the ConsoleTable environment that
again converts the generated tables to a static HTML page.
Below is an example outcome of `report.show(fb.export.HtmlTable)`.

![html_table.png](html_table.png)

## PlotlyHeatMap

!!! warning
    Plotly is installed as part of the interactive extras.

This is similar to the HtmlTable environment, with the 
difference that Plotly is used for heatmap plotting of
the values.


## ToJson

When provided as an environment to the `show` method, it does
not create any visual output but instead returns a string
that is a json damp of the report values. This can be used
to reconstruct the report like so:

```python
import fairbench as fb
import json

# serialize a report
json_dump = report.show(env=fb.export.ToJson)
# deserialize
dicts_and_lists = json.loads(json_dump)
reconstructed = fb.core.Value.from_dict(dicts_and_lists)
```
