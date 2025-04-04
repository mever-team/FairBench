# Report exploration

Here are described various ways to programmatically explore reports in
a simplified manner. These are different than just increasing the depth shown.
Some of these options are also used by interactive visualizations described
[here](../material/visualization.md).

## Specialize

Focus on certain aspects of reports based
on some key values. Keys any kind of computation name
included in the report, even those at high depth. 

For instance, write `report.samples.show()` to see the number of samples
involved in computations - this may be the same for all values of simple reports
but can be different if multiple reports are aggregated [here](progress.md).
All keys that may have been involved in computations are listed [here](../material/api.md).

Various keys are available, and obtained with `report.keys()`. These
are what FairBench internally calls *descriptors* and contain several
kinds of information, like a name, role, and details to describe. When 
printed, the name and role are shown. Here is an example:

```python
import fairbench as fb

sensitive = fb.Dimensions(men=[1, 1, 0, 0, 0], women=[0, 0, 1, 1, 1])
report = fb.reports.pairwise(
    predictions=[1, 0, 1, 0, 0], 
    labels=[1, 0, 0, 1, 0], 
    sensitive=sensitive
)

print(report.keys())
```

```text
[min [reduction], acc [measure], men [group], samples [count], ap [count], an [count], tp [count], tn [count], women [group], pr [measure], positives [count], tpr [measure], tnr [measure], negatives [count], tar [measure], trr [measure], max [reduction], maxerror [reduction], wmean [reduction], mean [reduction], maxrel [reduction], maxdiff [reduction], gini [reduction], std [reduction]]
```

The part in the bracket is the role, but you can use any key or its name
to specialize on the report. The dot notation shown above is one way to specialize 
given a name. Alternatively, you can access the specialization like a dictionary
based on its name or an item from the key list. This is mostly useful when there
are report entries (e.g., sensitive attribute dimensions) that contain spaces or
other special characters that Python cannot parse.
Here is an example, which also demonstrates that specializations can be chained:

```python
report["min"].acc.show() # equivalent to `report.min.acc.show()`
```

```text
##### min acc #####
|This reduction of a measure is the minimum of the accuracy.
|Value: 0.333 min acc

  (0.0, 1.0)
  ▎ █   
  ▎ █   
  ▎ █   
  ▎ █   
  ▎ █  █
  ▎▬*▬▬-
  (2.0, 0.0)
  
   * men                                 1 acc
   - women                               0.333 acc
```

## Explain

A special kind of specialization is the `explain` view, which can
be accessed only with the dot notation. This views inner details of the
top layer of report values, which may be equally instructive. The explain
view is different than just adding to the depth. 
Here is an example, where the explanation move from showing the minimum
accuracy distribution between men and women to comparing the quantities
that went into comparing them:

```python
report.min.acc.explain.show()
```

<div style="overflow-y: scroll;height: 380px; margin-bottom: 30px;">

```text
##### acc counts #####
|This measure of a view is the minimum of the accuracy across counts.
|Computations cover several cases.

 ***** samples *****
 |This is the sample count.
 |Computations cover several cases.
 
   (0.0, 3.0)
   ▎    █
   ▎    █
   ▎ █  █
   ▎ █  █
   ▎ █  █
   ▎▬*▬▬-
   (2.0, 0.0)
   
    * men                               2 samples
    - women                             3 samples
 
 ***** ap *****
 |This count is the actual positive labels.
 |Computations cover several cases.
 
   (0.0, 1.0)
   ▎ █  █
   ▎ █  █
   ▎ █  █
   ▎ █  █
   ▎ █  █
   ▎▬*▬▬-
   (2.0, 0.0)
   
    * men                               1 ap
    - women                             1 ap
 
 ***** an *****
 |This count is the actual negative labels.
 |Computations cover several cases.
 
   (0.0, 2.0)
   ▎    █
   ▎    █
   ▎    █
   ▎ █  █
   ▎ █  █
   ▎▬*▬▬-
   (2.0, 0.0)
   
    * men                               1 an
    - women                             2 an
 
 ***** tp *****
 |This count is the true positive predictions.
 |Computations cover several cases.
 
   (0.0, 1.0)
   ▎ █   
   ▎ █   
   ▎ █   
   ▎ █   
   ▎ █   
   ▎▬*▬▬-
   (2.0, 0.0)
   
    * men                               1 tp
    - women                             0 tp
 
 ***** tn *****
 |This count is the true negative predictions.
 |Computations cover several cases.
 
   (0.0, 1.0)
   ▎ █  █
   ▎ █  █
   ▎ █  █
   ▎ █  █
   ▎ █  █
   ▎▬*▬▬-
   (2.0, 0.0)
   
    * men                               1 tn
    - women                             1 tn
```

</div>

## Filters

Report outcomes have a `filter` method, in which
various kinds of investigative filters may be submitted.
Apply filters to focus on specific types of evaluation,
like keeping computations that show only bias
or keeping only bias/fairness values violating
certain thresholds. Find all available filters
[here](../material/filters.md).

One of the available filters, which is presented
below, are fairness stamps. These refer to a few 
common types of fairness evaluation and are accompanied
by caveats and recommendations. The collection of available
stamps is called a fairness modelcard, though it is
a normal report and can be manipulated (e.g., viewed) 
normally.

```python
report.filter(fb.investigate.Stamps).show(depth=1)
```


<div style="overflow-y: scroll;height: 380px; margin-bottom: 30px;">

```text
##### fairness modelcard #####
|This is a modelcard created with FairBench that consists of popular fairness 
|stamps.
|Stamps contain caveats and recommendation that should be considered during 
|practical adoption. They are only a part of the full analysis that has been 
|conducted, so consider also viewing the full generated report to find more 
|prospective biases.
|Computations cover several cases.

 ***** worst accuracy *****
 |This stamp is the minimum of the accuracy of analysis that compares several 
 |groups.
 |Value: 0.333 min acc
 
   ===== Details =====
   |This is the minimum benefit the system brings to any group.
   
   ===== Caveats and recommendations =====
   | • The worst case is a lower bound but not an estimation of overall 
   |   performance.
   | • There may be different distributions of benefits that could be 
   |   protected.
   | • Ensure continuous monitoring and re-evaluation as group dynamics and 
   |   external factors evolve.
   | • Ensure that high worst accuracy translates to meaningful benefits 
   |   across all groups in the real-world context.
   | • Seek input from affected groups to understand the impact of errors and 
   |   to inform remediation strategies.
   
   ===== Distribution =====
   
     (0.0, 1.0)
     ▎ █   
     ▎ █   
     ▎ █   
     ▎ █   
     ▎ █  █
     ▎▬*▬▬-
     (2.0, 0.0)
     
      * men                             1 acc
      - women                           0.333 acc
   
 ***** standard deviation *****
 |This stamp is the standard deviation of the accuracy of analysis that 
 |compares several groups.
 |Value: 0.333 
 
   ===== Details =====
   |This reflects imbalances in the distribution of benefits across groups.
   
   ===== Distribution =====
   
     (0.0, 1.0)
     ▎ █   
     ▎ █   
     ▎ █   
     ▎ █   
     ▎ █  █
     ▎▬*▬▬-
     (2.0, 0.0)
     
      * men                             1 acc
      - women                           0.333 acc
   
 ***** differential fairness *****
 |This stamp is the maximum relative difference of the accuracy of analysis 
 |that compares several groups.
 |Value: 0.667 
 
   ===== Details =====
   |The worst deviation of accuracy ratios from 1 is reported, so that value 
   |of 1 indicates disparate impact, and value of 0 disparate impact 
   |mitigation.
   
   ===== Caveats and recommendations =====
   | • Disparate impact may not always be an appropriate fairness 
   |   consideration, and may obscure other important fairness concerns or 
   |   create new disparities.
   | • Ensure continuous monitoring and re-evaluation as group dynamics and 
   |   external factors evolve.
   
   ===== Distribution =====
   
     (0.0, 1.0)
     ▎ █   
     ▎ █   
     ▎ █   
     ▎ █   
     ▎ █  █
     ▎▬*▬▬-
     (2.0, 0.0)
     
      * men                             1 acc
      - women                           0.333 acc
   
 ***** max |Δfpr| *****
 |This stamp is the maximum difference of the true negative rate of analysis 
 |that compares several groups.
 |Value: 0.500 
 
   ===== Details =====
   |The false positive rate differences are computed via the equivalent true 
   |negative rate differences. The maximum difference between pairs of groups 
   |is reported, so that value of 1 indicates disparate mistreatment, and 
   |value of 0 disparate mistreatment mitigation.
   
   ===== Caveats and recommendations =====
   | • Disparate mistreatment may not always be an appropriate fairness 
   |   consideration, and may obscure other important fairness concerns or 
   |   create new disparities.
   | • Consider input from affected stakeholders to determine whether |Δfpr| 
   |   is an appropriate fairness measure.
   | • Ensure continuous monitoring and re-evaluation as group dynamics and 
   |   external factors evolve.
   | • Variations in FPR could be influenced by factors unrelated to the 
   |   fairness of the system, such as data quality or representation.
   | • Mitigating |Δfpr| tends to mitigate |Δfnr|, and conversely.
   | • Seek input from affected groups to understand the impact of errors and 
   |   to inform remediation strategies.
   
   ===== Distribution =====
   
     (0.0, 1.0)
     ▎ █   
     ▎ █   
     ▎ █   
     ▎ █  █
     ▎ █  █
     ▎▬*▬▬-
     (2.0, 0.0)
     
      * men                             1 tnr
      - women                           0.500 tnr
   
 ***** max |Δfnr| *****
 |This stamp is the maximum difference of the true positive rate of analysis 
 |that compares several groups.
 |Value: 1.000 
 
   ===== Details =====
   |The false negative rate differences are computed via the equivalent true 
   |positive rate differences. The maximum difference between pairs of groups 
   |is reported, so that value of 1 indicates disparate mistreatment, and 
   |value of 0 disparate mistreatment mitigation.
   
   ===== Caveats and recommendations =====
   | • Disparate mistreatment may not always be an appropriate fairness 
   |   consideration, and may obscure other important fairness concerns or 
   |   create new disparities.
   | • Consider input from affected stakeholders to determine whether |Δfnr| 
   |   is an appropriate fairness measure.
   | • Ensure continuous monitoring and re-evaluation as group dynamics and 
   |   external factors evolve.
   | • Variations in FPR could be influenced by factors unrelated to the 
   |   fairness of the system, such as data quality or representation.
   | • Mitigating |Δfpr| tends to mitigate |Δfnr|, and conversely.
   | • Seek input from affected groups to understand the impact of errors and 
   |   to inform remediation strategies.
   
   ===== Distribution =====
   
     (0.0, 1.0)
     ▎ █   
     ▎ █   
     ▎ █   
     ▎ █   
     ▎ █   
     ▎▬*▬▬-
     (2.0, 0.0)
     
      * men                             1 tpr
      - women                           0 tpr
```
</div>