# Quick benchmarks

Here we show how to run and compare algorithms. To give a taste of
the library's ability to accommodate various kinds of data, we use computer vision examples.

Before starting, install FairBench with the extra dependencies (extras) required
to run benchmarks on models of the corresponding data types. 
Tabular data benchmarks only use implementations packed alongside the
library under its normal installation. To install with all extras run:

```shell
pip install --upgrade fairbench[graph,llm,vision]
```

!!! warning
    - Benchmarking standardization is a work in progress 
    and you may encounter interface breaking changes before FairBench v1.0.0.
    - Vision extras include pytorch,
    which downloads and stores several gigabytes of data.

## 1. Setup experiments

There is little distinction between running different algorithms 
or different data with the workflow shown next; each algorithm-data 
pair is distinct. This simplification is not mandatory, but simpler to explore
given that we aim to explore multiple types of bias assessment too.
Automatically download and run datasets and models with
FairBench, which is why we installed the extras installed above.
Alternatively, create predictions with your own workflow
and skip this step.

Datasets are set up as callable methods under `bench` 
modules; they run provided algorithms in
a standardized way. 
Datasets can also run with 
your own models. In domains where large
models are the norm, like Vision and LLMs, provided algorithms/models are assumed to be already trained.

Below is an example that runs two classifiers on the `utkface`
vision dataset. You can aso set up experiments where you pass a torch
model as a classifier instead of a string name.

```python
import fairbench as fb

experiments = {
    "flac utkface": lambda: fb.bench.vision.utkface(classifier="flac"),
    "badd utkface": lambda: fb.bench.vision.utkface(classifier="badd")
}
```

## 2. Gather reports

!!! tip
    Get familiar with generating standalone fairness reports in the [quickstart](quickstart.md). 

FairBench offers the `Progress` class to gather fairness reports
by registering them sequentially and can yield an amalgamation at any point.
The same class is also used for reports that show the evolution
of datasets and algorithms over time 
(both tracked progress and reports can be serialized to and from Json,
which allows for persistence, if needed).

Here is how to add report/value instances to progress and build
a report that contains all of them:

```python
settings = fb.Progress("settings")
for name, experiment in experiments.items():
    y, yhat, sensitive = experiment()
    sensitive = fb.Dimensions(fb.categories @ sensitive)
    report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
    settings.instance(name, report)
comparison = settings.build()

comparison.show(fb.export.ConsoleTable)
```

```text
                                    settings flac utkf  ace multidim                                                settings badd utkface multidim                                             
                                           acc           pr          tpr          tnr          tar          trr          acc           pr          tpr          tnr          tar          trr
min                                      0.903        0.417        0.859        0.921        0.382        0.474        0.904        0.431        0.901        0.866        0.401        0.446
max                                                   0.471                                  0.430        0.521                     0.527                                  0.458        0.525
maxerror                                 0.097                     0.141        0.079                                  0.096                     0.099        0.134                          
wmean                                    0.903        0.447        0.874        0.928        0.410        0.494        0.913        0.486        0.925        0.900        0.434        0.480
mean                                     0.903        0.444        0.872        0.930        0.406        0.497        0.915        0.479        0.922        0.906        0.430        0.485
maxrel                                   0.001        0.115        0.030        0.018        0.111        0.090        0.023        0.181        0.045        0.084        0.125        0.150
maxdiff                                  0.001        0.054        0.027        0.017        0.048        0.047        0.022        0.096        0.043        0.079        0.057        0.079
gini                                     0.000        0.030        0.008        0.005        0.029        0.024        0.006        0.050        0.012        0.022        0.033        0.041
std                                      0.000        0.027        0.013        0.009        0.024        0.023        0.011        0.048        0.021        0.040        0.029        0.039
```

## 3. Side-by-side comparison

Alter how reports are organized
at the top level using `.explain`. This is helpful for side-by-side comparison,
for example with the following recipe. The console table visualization
environment below requests usage `sideways=False` instead its default value once
it realizes there are conflicting row names, like *min acc*  vs *maxerror acc*. 
All visualizations generalize to any number of comparisons.

```python
comparison.explain.show(env=fb.export.ConsoleTable(sideways=False)) 
```

*Output omitted for brevity. It is similar to the last one in this page.*

## 4.Explore

Explore reports of any complexity by focusing on contributing
computations of interest. Do this programmatically with the dot notation or, 
when the former would not be valid Python, by looking up the computation as in
a dictionary with its string name.
Exploration is made easier with interactive visualization environments [here](documentation/interactive.md).
Or you can show all specification with the report's `.help()` method, which can be called
at any point of specialization.

Below is an exploration example. In this, we focus on both accuracy and the maximum difference reduction 
to keep the outcome simple.

```python
comparison.acc.explain["maxdiff explain mean"].show(env=fb.export.Console)
```

```text
##### maxdiff #####
|This reduction is the maximum difference.
|Computations cover several cases.

  (0.0, 0.02171521458324266)
  ▎    █
  ▎    █
  ▎    █
  ▎    █
  ▎    █
  ▎▬*▬▬-
  (2.0, 0.0)
  
   * flac utkface multidim               0.001 
   - badd utkface multidim               0.022 
```

## 5. Simplify the comparison 

Apply filters to retain only specific types of evaluation,
such as computations that show only bias
or bias/fairness values violating certain thresholds.

One of the available filters, which is presented
below, hides deviations from ideal values
lesser than `0.1` (also hides values whose
ideal targets cannot be determined). Observe
how the report is simplified into its most
problematic elements. If everything was rejected,
it would not indicate fairness -which is a subjective belief-
but rather that stricter thresholds should be explored.

```python
filter = fb.investigate.DeviationsOver(0.1)
env = fb.export.ConsoleTable(sideways=False) # not sideways because the environment complains about different rows
comparison.filter(filter).show(env=env)  
```

```text
settings flac utkface multidim
                                   maxerror       maxrel
tpr                                   0.141             
pr                                                 0.115
tar                                                0.111

settings badd utkface multidim
                                   maxerror       maxrel
tnr                                   0.134             
pr                                                 0.181
tar                                                0.125
trr                                                0.150
```

## 6. Multiple runs

Use the same progress tracking mechanism to keep track
of multiple experiment repetitions, if you need to. 
Then, apply the 
library's reduction mechanisms -the same ones employed
within reports- as filters to aggregate information
across repetitions.

In the example below, the average across 5 experiment
repetitions is shown for all measures. This just a demonstration, 
as the test classifiers
have no randomization and produce the same results always.
All computational information is still being tracked, so you can
delve into details afterwards.

```python
settings = fb.Progress("settings")
for name, experiment in experiments.items():
    repetitions = fb.Progress("5 repetitions")
    for repetition in range(5):
        y, yhat, sensitive = experiment()
        sensitive = fb.Dimensions(fb.categories @ sensitive)
        report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
        repetitions.instance(f"repetition {repetition}", report)
        
    # get the average across repetitions
    mean_report = repetitions.build().filter(fb.reduction.mean)
    settings.instance(name, mean_report)

comparison = settings.build()
comparison.explain.show(env=fb.export.ConsoleTable(sideways=False)) 
```

<div style="overflow-y: scroll;height: 380px; margin-bottom: 30px;">

```text
settings explain min explain mean
                                 flac utkface 5 repetitions   badd utkface 5 repetitions
mean acc                                              0.903                        0.904
mean pr                                               0.417                        0.431
mean tpr                                              0.859                        0.901
mean tnr                                              0.921                        0.866
mean tar                                              0.382                        0.401
mean trr                                              0.474                        0.446

settings explain max explain mean
                                 flac utkface 5 repetitions   badd utkface 5 repetitions
mean pr                                               0.471                        0.527
mean tar                                              0.430                        0.458
mean trr                                              0.521                        0.525

settings explain maxerror explain mean
                                 flac utkface 5 repetitions   badd utkface 5 repetitions
mean acc                                              0.097                        0.096
mean tpr                                              0.141                        0.099
mean tnr                                              0.079                        0.134

settings explain wmean explain mean
                                 flac utkface 5 repetitions   badd utkface 5 repetitions
mean acc                                              0.903                        0.913
mean pr                                               0.447                        0.486
mean tpr                                              0.874                        0.925
mean tnr                                              0.928                        0.900
mean tar                                              0.410                        0.434
mean trr                                              0.494                        0.480

settings explain mean explain mean
                                 flac utkface 5 repetitions   badd utkface 5 repetitions
mean acc                                              0.903                        0.915
mean pr                                               0.444                        0.479
mean tpr                                              0.872                        0.922
mean tnr                                              0.930                        0.906
mean tar                                              0.406                        0.430
mean trr                                              0.497                        0.485

settings explain maxrel explain mean
                                 flac utkface 5 repetitions   badd utkface 5 repetitions
mean acc                                              0.001                        0.023
mean pr                                               0.115                        0.181
mean tpr                                              0.030                        0.045
mean tnr                                              0.018                        0.084
mean tar                                              0.111                        0.125
mean trr                                              0.090                        0.150

settings explain maxdiff explain mean
                                 flac utkface 5 repetitions   badd utkface 5 repetitions
mean acc                                              0.001                        0.022
mean pr                                               0.054                        0.096
mean tpr                                              0.027                        0.043
mean tnr                                              0.017                        0.079
mean tar                                              0.048                        0.057
mean trr                                              0.047                        0.079

settings explain gini explain mean
                                 flac utkface 5 repetitions   badd utkface 5 repetitions
mean acc                                              0.000                        0.006
mean pr                                               0.030                        0.050
mean tpr                                              0.008                        0.012
mean tnr                                              0.005                        0.022
mean tar                                              0.029                        0.033
mean trr                                              0.024                        0.041

settings explain std explain mean
                                 flac utkface 5 repetitions   badd utkface 5 repetitions
mean acc                                              0.000                        0.011
mean pr                                               0.027                        0.048
mean tpr                                              0.013                        0.021
mean tnr                                              0.009                        0.040
mean tar                                              0.024                        0.029
mean trr                                              0.023                        0.039
```

</div>