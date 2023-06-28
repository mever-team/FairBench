## Factors
- The groups that are considered for fairness assessment are Female, Male.

## Metrics
- Fairness-aware metrics are computed. The *prule* compares the fraction of positive predictions between protected groups. The worst ratio between the groups is reported, so that value of 0 indicates indicates disparate impact, and value of 1 disparate impact mitigation. The *worst accuracy* computes the worst performance among protected groups; this is the minimum benefit the system brings to any group. The *80% rule* checks whether the fraction of positive predictions for each protected group is at worst 80% that of any other group. 

## Evaluation results
| Metric | Value |
| ------ | ----- |
| prule | 0.212 |
| worst accuracy | 0.799 |
| 80% rule | :x: |

## Caveats and Recommendations
- Consider input from affected stakeholders to determine whether the 80% rule is an appropriate fairness criterion.
- Disparate impact may not always be an appropriate fairness consideration.
- The worst accuracy is a lower bound but not an estimation of overall accuracy. There may be different distributions of benefits that could be protected.
