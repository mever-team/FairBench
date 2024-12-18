# Reports

Reports perform multi-faceted analyses of system outcomes
(e.g., predictions, recommendations, regression scores)
to quantify biases and fairness. Computations are retained
in the final report. Mainly, report outcomes can be filtered
to augment their values with various strategies,
and shown using various visualization environments.
They can also be serialized and deserialized.

## Arguments

You can generate reports by providing some
of the optional arguments below to some
of the report generation methods.
Base performance measures to compute
are automatically selected based on available information.

Sensitive attributes contain multiple
[dimensions](dimensions)
that treat multi-value attributes or multiple
sensitive attribute values interchangeably.

| Argument    | Role                | Values                                                         |
|-------------|---------------------|----------------------------------------------------------------|
| predictions | system output       | binary array                                                   |
| scores      | system output       | array with elements in [0,1]                                   |
| targets     | prediction target   | array with elements in [0,1]                                   |      
| labels      | prediction target   | binary array                                                   | 
| sensitive   | sensitive attribute | fork of arrays with elements in [0,1] (either binary or fuzzy) |

!!! info
    In multiclass settings, create the same 
    class dimensions on all provided keyword
    argument other than the sensitive attribute.

## Report types

Out-of-the box, you can use one of the following
report generation methods, obtained from FairBench's 
`reports`module:

| Report   | Description                                                                    |
|----------|--------------------------------------------------------------------------------|
| pairwise | Compares groups pairwise when needed.                                          |
| vsall    | Adds the whole population as an 'all' group and compares other groups to this. |
| report   | Creates a report with manually declared metrics and reduction strategies.      |


## Usage

As an example, create a simple report
based on binary predictions, binary
ideal predictions and multiclass
sensitive attribute `sensitive`. The
sensitive attribute is declared to be a fork with two branches
*men,women*, each of which is a binary
feature value. Finally, use the `show` method
to visualize the report using the console,
which is the default visualization engine.

```python
import fairbench as fb

sensitive = fb.Dimensions(men=[1, 1, 0, 0, 0], women=[0, 0, 1, 1, 1])
report = fb.reports.pairwise(
    predictions=[1, 0, 1, 0, 0], 
    labels=[1, 0, 0, 1, 0], 
    sensitive=sensitive
)
report.show()
```

Below is the visualization's outcome. Notice that descriptions
are provided for non-terminal values. 

<div style="overflow-y: scroll;height: 380px; margin-bottom: 30px;">

```
##### multidim #####
|This is analysis that compares several groups.
|Computations cover several cases.

 ***** min *****
 |This reduction is the minimum.
 |Computations cover several cases.
 
   (0.0, 0.5)
   ▎          █      
   ▎          █      
   ▎ █  █     █     █
   ▎ █  █     █     █
   ▎ █  █     █     █
   ▎▬*▬▬-▬▬+▬▬x▬▬o▬▬□
           (6.0, 0.0)
   
    * min acc                           0.333 
    - min pr                            0.333 
    + min tpr                           0 
    x min tnr                           0.500 
    o min tar                           0 
    □ min trr                           0.333 
 
 ***** max *****
 |This reduction is the maximum.
 |Computations cover several cases.
 
   (0.0, 0.5)
   ▎ █  █  █
   ▎ █  █  █
   ▎ █  █  █
   ▎ █  █  █
   ▎ █  █  █
   ▎▬*▬▬-▬▬+
   (3.0, 0.0)
   
    * max pr                            0.500 
    - max tar                           0.500 
    + max trr                           0.500 
 
 ***** maxerror *****
 |This reduction is the maximum deviation from the ideal value.
 |Computations cover several cases.
 
   (0.0, 1.0)
   ▎    █   
   ▎    █   
   ▎ █  █   
   ▎ █  █  █
   ▎ █  █  █
   ▎▬*▬▬-▬▬+
   (3.0, 0.0)
   
    * maxerror acc                      0.667 
    - maxerror tpr                      1 
    + maxerror tnr                      0.500 
 
 ***** wmean *****
 |This reduction is the weighted average.
 |Computations cover several cases.
 
   (0.0, 0.7)
   ▎          █      
   ▎ █        █      
   ▎ █        █      
   ▎ █  ▂  ▂  █     ▂
   ▎ █  █  █  █  ▄  █
   ▎▬*▬▬-▬▬+▬▬x▬▬o▬▬□
           (6.0, 0.0)
   
    * wmean acc                         0.600 
    - wmean pr                          0.400 
    + wmean tpr                         0.400 
    x wmean tnr                         0.700 
    o wmean tar                         0.200 
    □ wmean trr                         0.400 
 
 ***** mean *****
 |This reduction is the average.
 |Computations cover several cases.
 
   (0.0, 0.75)
   ▎          █      
   ▎ ▂        █      
   ▎ █     █  █      
   ▎ █  ▂  █  █     ▂
   ▎ █  █  █  █  █  █
   ▎▬*▬▬-▬▬+▬▬x▬▬o▬▬□
           (6.0, 0.0)
   
    * mean acc                          0.667 
    - mean pr                           0.417 
    + mean tpr                          0.500 
    x mean tnr                          0.750 
    o mean tar                          0.250 
    □ mean trr                          0.417 
 
 ***** maxrel *****
 |This reduction is the maximum relative difference.
 |Computations cover several cases.
 
   (0.0, 0.6666666666666667)
   ▎ █               
   ▎ █               
   ▎ █        ▂      
   ▎ █  █     █     █
   ▎ █  █     █     █
   ▎▬*▬▬-▬▬+▬▬x▬▬o▬▬□
           (6.0, 0.0)
   
    * maxrel acc                        0.667 
    - maxrel pr                         0.333 
    + maxrel tpr                        0 
    x maxrel tnr                        0.500 
    o maxrel tar                        0 
    □ maxrel trr                        0.333 
 
 ***** maxdiff *****
 |This reduction is the maximum difference.
 |Computations cover several cases.
 
   (0.0, 1.0)
   ▎       █         
   ▎       █         
   ▎ █     █         
   ▎ █     █  █  █   
   ▎ █     █  █  █   
   ▎▬*▬▬-▬▬+▬▬x▬▬o▬▬□
           (6.0, 0.0)
   
    * maxdiff acc                       0.667 
    - maxdiff pr                        0.167 
    + maxdiff tpr                       1 
    x maxdiff tnr                       0.500 
    o maxdiff tar                       0.500 
    □ maxdiff trr                       0.167 
 
 ***** gini *****
 |This reduction is the gini coefficient.
 |Computations cover several cases.
 
   (0.0, 0.5)
   ▎       █     █   
   ▎       █     █   
   ▎       █     █   
   ▎ █     █     █   
   ▎ █  █  █  █  █  █
   ▎▬*▬▬-▬▬+▬▬x▬▬o▬▬□
           (6.0, 0.0)
   
    * gini acc                          0.250 
    - gini pr                           0.100 
    + gini tpr                          0.500 
    x gini tnr                          0.167 
    o gini tar                          0.500 
    □ gini trr                          0.100 
 
 ***** std *****
 |This reduction is the standard deviation.
 |Computations cover several cases.
 
   (0.0, 0.5)
   ▎       █         
   ▎       █         
   ▎ █     █         
   ▎ █     █  █  █   
   ▎ █     █  █  █   
   ▎▬*▬▬-▬▬+▬▬x▬▬o▬▬□
           (6.0, 0.0)
   
    * std acc                           0.333 
    - std pr                            0.083 
    + std tpr                           0.500 
    x std tnr                           0.250 
    o std tar                           0.250 
    □ std trr                           0.083 
```
</div>

## Visualization

The above example uses the `show` method to print to the console.
The same method can create various types of exports.
For example, view the report's outcome in the browser by using a different visualization
argument, such as with `report.show(env=fb.export.Html)`. Find a 
full list of visualization environments [here](../material/visualization.md).

The same method also admits a `depth` argument that determines
the level of detail, where the default is zero. At least the top
set of numeric values will be shown regardless of the detail level.
All visualization environments can work with any set depth, though
beware that large depths may create imprtactically many details;
you might want to specialize like below.
