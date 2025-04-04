# Report keys

Here is a list of all keys that you may try to extract from reports,
for example with the dot notation like `report.acc` to obtain
all accuracy computations or `report.min.roc` to obtain all receiver operating
characteristic curves under minimum reductions. Not all keys are present in all reports.

!!! info
    This is a reminder that
    viewing reports based on keys comes in addition to user-defined sensitive attribute dimension names and 
    `Progress` instance names. For example, use `report["male"]` or `report.male` to
    focus on the namesake sensitive attribute dimension - the dict notation is a supplement
    in case the names have spaces or special characters.

## Quantities

These are raw quantities that are contained in measures or reductions.

| Name          | Role       | Details                                                |
|---------------|------------|--------------------------------------------------------|
| samples       | count      | the sample count                                       |
| positives     | count      | the positive predictions                               |
| negatives     | count      | the negative predictions                               |
| tp            | count      | the true positive predictions                          |
| tn            | count      | the true negative predictions                          |
| ap            | count      | the actual positive labels                             |
| an            | count      | the actual negative labels                             |
| freedom       | parameter  | the degrees of freedom                                 |
| slope         | parameter  | the slope of pinball deviation                         |
| distribution  | curve      | the score distribution                                 |
| roc           | curve      | the receiver operating characteristics curve           |
| top           | count      | the number of top scores considered                    |
| precision     | metric     | the precision score                                    |
| repr          | metric     | the representation in top samples                      |


## Measures

These are measures of algorithmic performance.

| Name     | Type & role            | Details                                                        |
|----------|------------------------|----------------------------------------------------------------|
| pr       | cLassification measure | the positive rate                                              |
| acc      | cLassification measure | the accuracy                                                   |
| tpr      | cLassification measure | the true positive rate                                         |
| tnr      | cLassification measure | the true negative rate                                         |
| tar      | cLassification measure | the true acceptance rate                                       |
| trr      | cLassification measure | the true rejection rate                                        |
| avgscore | ranking measure        | the average score                                              |
| auc      | ranking measure        | the area under curve of the receiver operating characteristics |
| tophr    | ranking measure        | the hit ratio of top recommendations                           |
| toprec   | ranking measure        | the precision of top recommendations                           |
| topf1    | ranking measure        | the F1 score of top recommendations                            |
| avgrepr  | ranking measure        | the average representation at top recommendations              |
| mabs     | regression measure     | mean absolute error                                            |
| rmse     | regression measure     | root mean square error                                         |
| mse      | regression measure     | mean square error                                              |
| r2       | regression measure     | coefficient of determination                                   |
| pinball  | regression measure     | pinball deviation                                              |

## Reductions

| Name             | Role       | Details                                                                                               |
|------------------|------------|-------------------------------------------------------------------------------------------------------|
| min              | reduction  | the minimum                                                                                           |
| max              | reduction  | the maximum                                                                                           |
| maxerror         | reduction  | the maximum deviation from the ideal value                                                            |
| std              | reduction  | the standard deviation                                                                                |
| gini             | reduction  | the gini coefficient                                                                                  |
| mean             | reduction  | the average                                                                                           |
| wmean            | reduction  | the weighted average                                                                                  |
| maxdiff          | reduction  | the maximum difference                                                                                |
| maxbarea         | reduction  | the maximum area between curves                                                                       |
| maxrel           | reduction  | the maximum relative difference                                                                       |
| largestmaxdiff   | reduction  | the maximum difference from the largest group (the whole population if included)                      |
| largestmaxrel    | reduction  | the maximum relative difference from the largest group (the whole population if included)             |
| largestmaxbarea  | reduction  | the maximum area between curves and the curve of the largest group (the whole population if included) |

