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

To assess your system, start by generating predictions for test data.
Here we evaluate a binary classifier that is available for out-of-the-box
experimentation; outputs are the test portion of the dataset *x*, 
the target binary labels *y*, and binary predicted labels *yhat*.
We are interested in seeing how fair those predictions are.

```python
import fairbench as fb
test, y, yhat = fb.bench.tabular.adult()
```

## 2. Sensitive attributes

Pack sensitive attributes found across test samples
into a data structure holding multiple [dimensions](documentation/dimensions.md).
This stores any number of attributes with any number of values
by considering each value as a separate dimension.

In particular, each dimension is a binary or fuzzy array
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
with reduction strategies (columns). For example, the value at
column `min` and row `acc` indicates the minimum accuracy 
across all sensitive groups. Similarly, the value at column maximum
relative difference `maxrel` and row positive rate `pr`
indicate the maximum relative difference between the positive
rates of all groups; this is the differential fairness
extension to the well-known prule measure originally
coined for binary sensitive attributes (see below).

The task type and applicable measures are determined
by the report's arguments. 
See the comprehensive list of all 
[measures and reductions](material/api.md), 
but you can find out what the abbreviations refer
to by either using a visualization environment
that prints such information (see next section)
or simply calling `report.help()` to view lengthier
descriptions of all factors contributing to
the generated report.

```python
report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
report.show(env=fb.export.ConsoleTable)  
report.help()
```

This is the outcome of `show`:

<pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">                                                                                                                                       
                                           min          max     maxerror        wmean         mean           gm        pnorm       maxrel      maxdiff         gini        stdx2
acc                                      <span style="color:#5a9e6f">0.959</span>                     <span style="color:#5a9e6f">0.041</span>        <span style="color:#5a9e6f">0.968</span>        <span style="color:#5a9e6f">0.978</span>        <span style="color:#5a9e6f">0.978</span>            <span style="color:#c75f5f">3</span>        <span style="color:#5a9e6f">0.041</span>        <span style="color:#5a9e6f">0.041</span>        <span style="color:#5a9e6f">0.008</span>        <span style="color:#5a9e6f">0.027</span>
pr                                       <span style="color:#b8956a">0.241</span>        <span style="color:#b8956a">0.667</span>                     <span style="color:#b8956a">0.475</span>        <span style="color:#b8956a">0.442</span>        <span style="color:#b8956a">0.424</span>            <span style="color:#b8956a">1</span>        <span style="color:#b8956a">0.638</span>        <span style="color:#b8956a">0.425</span>        <span style="color:#5a9e6f">0.157</span>        <span style="color:#b8956a">0.250</span>
tpr                                          <span style="color:#5a9e6f">1</span>            <span style="color:#5a9e6f">1</span>            <span style="color:#5a9e6f">0</span>            <span style="color:#5a9e6f">1</span>            <span style="color:#5a9e6f">1</span>            <span style="color:#5a9e6f">1</span>            <span style="color:#c75f5f">3</span>            <span style="color:#5a9e6f">0</span>            <span style="color:#5a9e6f">0</span>            <span style="color:#5a9e6f">0</span>            <span style="color:#5a9e6f">0</span>
tnr                                      <span style="color:#5a9e6f">0.912</span>                     <span style="color:#5a9e6f">0.088</span>        <span style="color:#5a9e6f">0.939</span>        <span style="color:#5a9e6f">0.964</span>        <span style="color:#5a9e6f">0.964</span>            <span style="color:#c75f5f">3</span>        <span style="color:#5a9e6f">0.088</span>        <span style="color:#5a9e6f">0.088</span>        <span style="color:#5a9e6f">0.014</span>        <span style="color:#5a9e6f">0.050</span>
ppv                                      <span style="color:#5a9e6f">0.857</span>                     <span style="color:#5a9e6f">0.143</span>        <span style="color:#5a9e6f">0.932</span>        <span style="color:#5a9e6f">0.942</span>        <span style="color:#5a9e6f">0.942</span>            <span style="color:#c75f5f">2</span>        <span style="color:#5a9e6f">0.143</span>        <span style="color:#5a9e6f">0.143</span>        <span style="color:#5a9e6f">0.022</span>        <span style="color:#5a9e6f">0.078</span>
f1                                       <span style="color:#5a9e6f">0.923</span>                     <span style="color:#5a9e6f">0.077</span>                     <span style="color:#5a9e6f">0.970</span>        <span style="color:#5a9e6f">0.970</span>            <span style="color:#c75f5f">3</span>        <span style="color:#5a9e6f">0.077</span>        <span style="color:#5a9e6f">0.077</span>        <span style="color:#5a9e6f">0.011</span>        <span style="color:#5a9e6f">0.042</span>
gmi                                      <span style="color:#5a9e6f">0.926</span>                     <span style="color:#5a9e6f">0.074</span>                     <span style="color:#5a9e6f">0.971</span>        <span style="color:#5a9e6f">0.970</span>            <span style="color:#c75f5f">3</span>        <span style="color:#5a9e6f">0.074</span>        <span style="color:#5a9e6f">0.074</span>        <span style="color:#5a9e6f">0.011</span>        <span style="color:#5a9e6f">0.040</span>
tar                                      <span style="color:#b8956a">0.207</span>        <span style="color:#b8956a">0.667</span>                     <span style="color:#b8956a">0.443</span>        <span style="color:#b8956a">0.420</span>        <span style="color:#b8956a">0.399</span>            <span style="color:#b8956a">1</span>        <span style="color:#b8956a">0.690</span>        <span style="color:#b8956a">0.460</span>        <span style="color:#5a9e6f">0.172</span>        <span style="color:#b8956a">0.264</span>
trr                                      <span style="color:#b8956a">0.333</span>        <span style="color:#b8956a">0.759</span>                     <span style="color:#b8956a">0.525</span>        <span style="color:#b8956a">0.558</span>        <span style="color:#b8956a">0.543</span>            <span style="color:#b8956a">1</span>        <span style="color:#b8956a">0.561</span>        <span style="color:#b8956a">0.425</span>        <span style="color:#5a9e6f">0.124</span>        <span style="color:#b8956a">0.250</span>
lift                                         <span style="color:#b8956a">1</span>            <span style="color:#b8956a">4</span>                         <span style="color:#b8956a">2</span>            <span style="color:#b8956a">2</span>            <span style="color:#b8956a">2</span>            <span style="color:#b8956a">8</span>        <span style="color:#b8956a">0.638</span>            <span style="color:#c75f5f">2</span>        <span style="color:#5a9e6f">0.159</span>            <span style="color:#c75f5f">1</span>
mcc                                      <span style="color:#5a9e6f">0.905</span>                     <span style="color:#5a9e6f">0.095</span>                     <span style="color:#5a9e6f">0.953</span>        <span style="color:#5a9e6f">0.953</span>            <span style="color:#c75f5f">3</span>        <span style="color:#5a9e6f">0.095</span>        <span style="color:#5a9e6f">0.095</span>        <span style="color:#5a9e6f">0.017</span>        <span style="color:#5a9e6f">0.059</span>
kappa                                    <span style="color:#5a9e6f">0.901</span>                     <span style="color:#5a9e6f">0.099</span>        <span style="color:#5a9e6f">0.933</span>        <span style="color:#5a9e6f">0.952</span>        <span style="color:#5a9e6f">0.951</span>            <span style="color:#c75f5f">3</span>        <span style="color:#5a9e6f">0.099</span>        <span style="color:#5a9e6f">0.099</span>        <span style="color:#5a9e6f">0.018</span>        <span style="color:#5a9e6f">0.061</span>
</pre>

This is the outcome of `help`:

<pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">
<span style="color:#5a9e6f">##### FairBench help #####</span>
Access the following fields of the selected value to explore results:
- <span style="color:#5f82c7">value.min                </span> This reduction<span style="color:#7fbf8f"> is </span>the minimum.
- <span style="color:#5f82c7">value.acc                </span> This measure<span style="color:#7fbf8f"> is </span>the accuracy.
- <span style="color:#5f82c7">value['Female&African-American']</span> This<span style="color:#7fbf8f"> is </span>the value<span style="color:#7fbf8f"> for </span>group 'Female&African-American'.
- <span style="color:#5f82c7">value.samples            </span> This<span style="color:#7fbf8f"> is </span>the sample count.
- <span style="color:#5f82c7">value.ap                 </span> This count<span style="color:#7fbf8f"> is </span>the actual positive labels.
- <span style="color:#5f82c7">value.an                 </span> This count<span style="color:#7fbf8f"> is </span>the actual negative labels.
- <span style="color:#5f82c7">value.tp                 </span> This count<span style="color:#7fbf8f"> is </span>the true positive predictions.
- <span style="color:#5f82c7">value.tn                 </span> This count<span style="color:#7fbf8f"> is </span>the true negative predictions.
- <span style="color:#5f82c7">value['Female&Hispanic'] </span> This<span style="color:#7fbf8f"> is </span>the value<span style="color:#7fbf8f"> for </span>group 'Female&Hispanic'.
- <span style="color:#5f82c7">value['Female&Other']    </span> This<span style="color:#7fbf8f"> is </span>the value<span style="color:#7fbf8f"> for </span>group 'Female&Other'.
- <span style="color:#5f82c7">value['Female&Caucasian']</span> This<span style="color:#7fbf8f"> is </span>the value<span style="color:#7fbf8f"> for </span>group 'Female&Caucasian'.
- <span style="color:#5f82c7">value['Female&Native American']</span> This<span style="color:#7fbf8f"> is </span>the value<span style="color:#7fbf8f"> for </span>group 'Female&Native American'.
- <span style="color:#5f82c7">value['Male&African-American']</span> This<span style="color:#7fbf8f"> is </span>the value<span style="color:#7fbf8f"> for </span>group 'Male&African-American'.
- <span style="color:#5f82c7">value['Male&Hispanic']   </span> This<span style="color:#7fbf8f"> is </span>the value<span style="color:#7fbf8f"> for </span>group 'Male&Hispanic'.
- <span style="color:#5f82c7">value['Male&Other']      </span> This<span style="color:#7fbf8f"> is </span>the value<span style="color:#7fbf8f"> for </span>group 'Male&Other'.
- <span style="color:#5f82c7">value['Male&Caucasian']  </span> This<span style="color:#7fbf8f"> is </span>the value<span style="color:#7fbf8f"> for </span>group 'Male&Caucasian'.
- <span style="color:#5f82c7">value['Male&Native American']</span> This<span style="color:#7fbf8f"> is </span>the value<span style="color:#7fbf8f"> for </span>group 'Male&Native American'.
- <span style="color:#5f82c7">value.pr                 </span> This measure<span style="color:#7fbf8f"> is </span>the positive rate.
- <span style="color:#5f82c7">value.positives          </span> This count<span style="color:#7fbf8f"> is </span>the positive predictions.
- <span style="color:#5f82c7">value.tpr                </span> This measure<span style="color:#7fbf8f"> is </span>the true positive rate/recall/sensitivity/hit rate.
- <span style="color:#5f82c7">value.tnr                </span> This measure<span style="color:#7fbf8f"> is </span>the true negative rate/specificity.
- <span style="color:#5f82c7">value.negatives          </span> This count<span style="color:#7fbf8f"> is </span>the negative predictions.
- <span style="color:#5f82c7">value.ppv                </span> This measure<span style="color:#7fbf8f"> is </span>the positive predictive value/precision.
- <span style="color:#5f82c7">value.f1                 </span> This measure<span style="color:#7fbf8f"> is </span>the f1 score.
- <span style="color:#5f82c7">value.gmi                </span> This measure<span style="color:#7fbf8f"> is </span>the geometric mean<span style="color:#7fbf8f"> of </span>tpr and tnr - accounts<span style="color:#7fbf8f"> for </span>class 
|imbalance.
- <span style="color:#5f82c7">value.tar                </span> This measure<span style="color:#7fbf8f"> is </span>the true acceptance ratio (true positives compared to all).
- <span style="color:#5f82c7">value.trr                </span> This measure<span style="color:#7fbf8f"> is </span>the true rejection ratio (true negatives compared to all).
- <span style="color:#5f82c7">value.lift               </span> This measure<span style="color:#7fbf8f"> is </span>the lift ratio (tpr divided by pr).
- <span style="color:#5f82c7">value.mcc                </span> This measure<span style="color:#7fbf8f"> is </span>the Matthews correlation coefficient.
- <span style="color:#5f82c7">value.kappa              </span> This measure<span style="color:#7fbf8f"> is </span>the Cohen's Kappa score.
- <span style="color:#5f82c7">value.max                </span> This reduction<span style="color:#7fbf8f"> is </span>the maximum.
- <span style="color:#5f82c7">value.maxerror           </span> This reduction<span style="color:#7fbf8f"> is </span>the maximum deviation from the ideal value.
- <span style="color:#5f82c7">value.wmean              </span> This reduction<span style="color:#7fbf8f"> is </span>the weighted average.
- <span style="color:#5f82c7">value.mean               </span> This reduction<span style="color:#7fbf8f"> is </span>the average.
- <span style="color:#5f82c7">value.gm                 </span> This reduction<span style="color:#7fbf8f"> is </span>the geometric mean.
- <span style="color:#5f82c7">value.pnorm              </span> This reduction<span style="color:#7fbf8f"> is </span>the p-norm (default L2).
- <span style="color:#5f82c7">value.maxrel             </span> This reduction<span style="color:#7fbf8f"> is </span>the maximum relative difference.
- <span style="color:#5f82c7">value.maxdiff            </span> This reduction<span style="color:#7fbf8f"> is </span>the maximum difference.
- <span style="color:#5f82c7">value.gini               </span> This reduction<span style="color:#7fbf8f"> is </span>the gini coefficient.
- <span style="color:#5f82c7">value.stdx2              </span> This reduction<span style="color:#7fbf8f"> is </span>the standard deviation x2.
</pre>

## 4. Go into details

Explore reports by focusing on any of their contributing
computations. Use the programmatic dot notation,
where more `depth` can be added to viewed values
to further expand intermediate computations and search
for the root causes of discrimination. You can 
focus on any of the factors shown via `report.help()`,
and even chain multiple specializations. Use dictionary
access notation if the dot notation would be invalid
(e.g., when the specialization includes spaces or special
characters). 

Below is an example, but there are 
[more visualization options](material/visualization.md).
In the example, we focus on the minimum accuracy to keep the 
outcome simple, but all visualization environments
work with full reports, or even 
[collections of reports](documentation/progress.md).

```python
report.min.acc.show(env=fb.export.Console)
```

This is the console output:

<pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">
<span style="color:#5f82c7">##### acc #####</span>
|This measure<span style="color:#7fbf8f"> is </span>the minimum<span style="color:#7fbf8f"> of </span>the accuracy.
|Value:<span style="color:#5a9e6f"> 0.929 min acc</span>

  (0.0, 1.0)
  ▎    <span style="color:#7fbf8f">█  </span><span style="color:#c8aa7a">█  </span><span style="color:#7f9fd4">█           </span><span style="color:#7fc0ca">█            
  </span>▎ <span style="color:#d47f7f">▆  </span><span style="color:#7fbf8f">█  </span><span style="color:#c8aa7a">█  </span><span style="color:#7f9fd4">█  </span><span style="color:#b47fc4">▆  </span><span style="color:#7fc0ca">▆  </span><span style="color:#c8aa7a">▆  </span><span style="color:#7fc0ca">█  </span><span style="color:#d47f7f">▆  </span><span style="color:#7fbf8f">▄  </span><span style="color:#7f9fd4">▆  </span><span style="color:#b47fc4">▆
  </span>▎ <span style="color:#d47f7f">█  </span><span style="color:#7fbf8f">█  </span><span style="color:#c8aa7a">█  </span><span style="color:#7f9fd4">█  </span><span style="color:#b47fc4">█  </span><span style="color:#7fc0ca">█  </span><span style="color:#c8aa7a">█  </span><span style="color:#7fc0ca">█  </span><span style="color:#d47f7f">█  </span><span style="color:#7fbf8f">█  </span><span style="color:#7f9fd4">█  </span><span style="color:#b47fc4">█
  </span>▎ <span style="color:#d47f7f">█  </span><span style="color:#7fbf8f">█  </span><span style="color:#c8aa7a">█  </span><span style="color:#7f9fd4">█  </span><span style="color:#b47fc4">█  </span><span style="color:#7fc0ca">█  </span><span style="color:#c8aa7a">█  </span><span style="color:#7fc0ca">█  </span><span style="color:#d47f7f">█  </span><span style="color:#7fbf8f">█  </span><span style="color:#7f9fd4">█  </span><span style="color:#b47fc4">█
  </span>▎ <span style="color:#d47f7f">█  </span><span style="color:#7fbf8f">█  </span><span style="color:#c8aa7a">█  </span><span style="color:#7f9fd4">█  </span><span style="color:#b47fc4">█  </span><span style="color:#7fc0ca">█  </span><span style="color:#c8aa7a">█  </span><span style="color:#7fc0ca">█  </span><span style="color:#d47f7f">█  </span><span style="color:#7fbf8f">█  </span><span style="color:#7f9fd4">█  </span><span style="color:#b47fc4">█
  </span>▎▬<span style="color:#d47f7f">*</span>▬▬<span style="color:#7fbf8f">-</span>▬▬<span style="color:#c8aa7a">+</span>▬▬<span style="color:#7f9fd4">x</span>▬▬<span style="color:#b47fc4">o</span>▬▬<span style="color:#7fc0ca">□</span>▬▬<span style="color:#c8aa7a">◇</span>▬▬<span style="color:#7fc0ca">#</span>▬▬<span style="color:#d47f7f">@</span>▬▬<span style="color:#7fbf8f">%</span>▬▬<span style="color:#7f9fd4">&</span>▬▬<span style="color:#b47fc4">|</span>
                           (12.0, 0.0)
  
   <span style="color:#d47f7f">* </span>Female&Caucasian                    0.978 acc
   <span style="color:#7fbf8f">- </span>Female&Native American              1 acc
   <span style="color:#c8aa7a">+ </span>Female&Other                        1 acc
   <span style="color:#7f9fd4">x </span>Female&Asian                        1 acc
   <span style="color:#b47fc4">o </span>Female&African-American             0.985 acc
   <span style="color:#7fc0ca">□ </span>Female&Hispanic                     0.959 acc
   <span style="color:#c8aa7a">◇ </span>Male&Caucasian                      0.971 acc
   <span style="color:#7fc0ca"># </span>Male&Native American                1 acc
   <span style="color:#d47f7f">@ </span>Male&Other                          0.973 acc
   <span style="color:#7fbf8f">% </span>Male&Asian                          0.929 acc
   <span style="color:#7f9fd4">& </span>Male&African-American               0.961 acc
   <span style="color:#b47fc4">| </span>Male&Hispanic                       0.981 acc
</pre>

## 5. Simplify reports 

FairBench provides filters that enhance the evaluation.
For example, they can remove report entries that do not violate a given
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
The provided depth controls the level of detail. Default depth
is zero, which automatically selects the shallowest meaningful
level of detail.

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

