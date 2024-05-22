# Reports

Reports perform multi-faceted analyses of system outcomes
(e.g., predictions, recommendations, regression scores)
and produce explainable high-level views of biases found
across several definitions of fairness. For interpretation
of what report outcomes mean, 
look at the <sub><sup>REPORT ENTRIES</sup></sub> section.


!!! tip
    To check for some well-known fairness metric, 
    produce reports with
    relevant information (e.g., multireports shown below)
    and extract the metric with its [stamp](../advanced/modelcards.md#stamps).

## Arguments

You can generate fairness reports by providing some
of the optional arguments below to a report
generation method. These arguments are needed to automatically
understand which base
performance [metrics](../record/metrics.md) to compute.
Report generation methods will try to compute fairness
assessment built from as many base metrics as they can,
depending on which arguments are provided.
Sensitive attributes are [forks](forks.md)
to handle multi-value attributes or multiple
sensitive attribute values.

| Argument    | Role                | Values                                                         |
|-------------|---------------------|----------------------------------------------------------------|
| predictions | system output       | binary array                                                   |
| scores      | system output       | array with elements in [0,1]                                   |
| targets     | prediction target   | array with elements in [0,1]                                   |      
| labels      | prediction target   | binary array                                                   | 
| sensitive   | sensitive attribute | fork of arrays with elements in [0,1] (either binary or fuzzy) |

!!! info
    In multiclass settings, create a report for
    each class label and combine them. See [here](interactive.md).

## Report types

Out-of-the box, you can use one of the following
report generation methods:

| Report      | Description                                                                                   | Best for                                                                     |
|-------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| accreport   | Provides popular performance evaluation measures to be viewed between branches.               | Just metrics: accuracy, positives, true positive rates, true negative rates. |
| binreport   | Conducts a suite of popular binary fairness assessments on each variable branch.               | Branches that do *not* correspond to sensitive attributes.                   |
| multireport | Ideal for multi-fairness approaches, where the `sensitive` fork has many branches.             | Multidimensional analysis                                                    |
| unireport   | Similar to `multireport`, but each group or subgroup is compared to the whole population.      | Group-vs-population comparisons.                                             |
| isecreport  | Tackles multi-fairness with many intersectional groups.                                        | Bayesian analysis for small or empty intersections.                          |

As an example, let's create a simple report
based on binary predictions, binary
ideal predictions and multiclass
sensitive attribute `sensitive`. The
sensitive attribute is
declared to be a fork with two branches
*men,women*, each of which is a binary
feature value. The report looks like this
when displayed:

```python
import fairbench as fb
sensitive = fb.Fork(men=[1, 1, 0, 0, 0], women=[0, 0, 1, 1, 1])
report = fb.multireport(
    predictions=[1, 0, 1, 0, 0], 
    labels=[1, 0, 0, 1, 0], 
    sensitive=sensitive)
```


<button onclick="toggleCode('print')" class="toggle-button">>></button>
Printing a report creates a yaml representation
of its contents. Use this only for a quick
peek, as there are better ways to explore reports,
described below. Entries that can not be computed
given the provided arguments are not filled.

<div id="print" class="code-block" style="display:none;">

```python
print(report)
```

```
min: 
   accuracy: 0.333
   pr: 0.333
   tpr: 0.000
   tnr: 0.500
wmean: 
   accuracy: 0.600
   pr: 0.400
   tpr: 0.400
   tnr: 0.700
gini: 
   accuracy: 0.250
   pr: 0.100
   tpr: 0.500
   tnr: 0.167
minratio: 
   accuracy: 0.333
   pr: 0.667
   tpr: 0.000
   tnr: 0.500
maxdiff: 
   accuracy: 0.667
   pr: 0.167
   tpr: 1.000
   tnr: 0.500
maxbarea: 
   accuracy: ---
   pr: ---
   tpr: ---
   tnr: ---
maxrarea: 
   accuracy: ---
   pr: ---
   tpr: ---
   tnr: ---
maxbdcg: 
   accuracy: ---
   pr: ---
   tpr: ---
   tnr: ---
```

</div>


<button onclick="toggleCode('metricsarg')" class="toggle-button">>></button>
You can restrict which metrics
the report considers by providing a `metrics`
argument. This should hold either 
a dictionary mapping names to metrics
or a list of metrics, where
in the last case their names are automatically inferred.
You can also add customly defined metrics.

<div id="metricsarg" class="code-block" style="display:none;">

```python
import fairbench as fb
report = fb.accreport( # just print performance metrics
    predictions=[1, 0, 1, 0, 0], 
    labels=[1, 0, 0, 1, 0], 
    metrics=[fb.accuracy, fb.pr, fb.fpr, fb.fnr])
fb.describe(report)  # pretty print - more on this later
```

</div>

## Show reports

Reports are forks whose branches hold dictionaries of
metric computations. In some reports (e.g., multireport
and unireport) [reduction](../advanced/manipulation.md)
operations introduce a comparative analysis
between the sensitive attribute branches to investigate
unfairness. For example, `min` shows the worst evaluation
across sensitive groups, 
and `minratio` and `maxdiff` the minimum ratio
and maximum differences between metric values for 
sensitive groups (groups correspond to sensitive 
attribute branches). Some comparison mechanisms
may also consider intermediate computations, like
distributions, used to compute the metrics.
Given a report, you can find what its comparisons mean
[here](../record/comparisons.md).

Several methods are provided to
work with the report data format, namely 
forks of dictionaries. First, you can show 
reports in the `stdout` console in the form
of tables:

```python
fb.describe(report)  

Metric          min             minratio        maxdiff        
accuracy        0.938           1.000           0.000          
pr              0.812           0.857           0.125          
fpr             0.063           0.778           0.016          
fnr             0.333           0.333           0.333  
```


<button onclick="toggleCode('latex')" class="toggle-button">>></button>
You can use the arguments to make `describe`
return a string without printing, or to create
latex tables.

<div id="latex" class="code-block" style="display:none;">

```python
import fairbench as fb

test, y, yhat = fb.demos.adult(predict="probabilities")
s = fb.Fork(fb.categories @ test[9])
report = fb.unireport(scores=yhat, labels=y, sensitive=s)

text = fb.describe(report,
    show=False, # prevents immediate printing
    separator=" & ", # separator between columns
    newline="\\\\\n") # use \\ and then the newline character
print(text)

Metric          & min             & wmean           & gini            & minratio[vsAny] & maxdiff[vsAny]  & maxbarea[vsAny] & maxrarea[vsAny] & maxbdcg[vsAny] \\
auc             & 0.861           & 0.882           & 0.012           & 0.972           & 0.025           & 0.025           & 0.038           & 0.028          \\
avgscore        & 0.110           & 0.239           & 0.197           & 0.461           & 0.129           & 0.454           & 0.548           & 0.499          \\
tophr           & 0.667           & 0.722           & 0.095           & 1.000           & 0.333           & nan             & nan             & nan            \\
toprec          & 0.001           & 0.001           & 0.489           & 1.181           & 0.005           & nan             & nan             & nan            \\
avghr           & 0.389           & 0.491           & 0.229           & 1.000           & 0.611           & 0.611           & 0.611           & 0.696          \\
avgrepr         & 0.000           & 1.000           & 0.400           & 0.000           & 1.000           & 1.000           & 1.000           & 1.000          \\
```


</div>


<button onclick="toggleCode('json')" class="toggle-button">>></button>
You can convert reports to *json*, for example 
to send to some frontend:

<div id="json" class="code-block" style="display:none;">

```python
print(fb.tojson(report))

{"header": ["Metric", "mean", "minratio", "maxdiff"], "accuracy": [0.9375, 1.0, 0.0], "pr": [0.8125, 0.8571428571428571, 0.125], "fpr": [0.06349206349206349, 0.7777777777777778, 0.015873015873015872], "fnr": [0.3333333333333333, 0.3333333333333333, 0.33333333333333337]}
```

</div>

## Plotting

You can plot reports with the command `fb.visualize(report)`.


```python
fb.visualize(report)
```
<img src="../reports.png" alt="report example">


<button onclick="toggleCode('matplotlib')" class="toggle-button">>></button>
Visualization uses `matplotlib.pyplot` under the hood. You can
rotate the horizontal axis labels, for example
when there are too many sensitive attribute dimensions,
and can ask for the visualization not to open the figure
so that you can perform additional operations.

<div id="matplotlib" class="code-block" style="display:none;">

```python
import matplotlib.pyplot as plt
import fairbench as fb
report = ...  # generate the report
fb.visualize(report, 
             xrotation=90, # rotate x labels
             hold=True)
plt.title("Report")
plt.show() # only show now
```

</div>

!!! warning 
    Complicated forks (e.g., forks of reports)
    cannot be displayed or visualized.
    But they can be converted to strings, printed, 
    or [interacted](interactive.md) with.




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