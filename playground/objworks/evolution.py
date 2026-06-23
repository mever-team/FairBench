import fairbench as fb
import numpy as np

x, y, yhat = fb.bench.tabular.bank(predict="probabilities")
sensitive = fb.Dimensions(fb.categories @ x["marital"])
print(sensitive)

comparison = fb.Progress("thresholds")
for threshold in np.arange(0.1, 0.91, 0.1):
    report = fb.reports.pairwise(
        sensitive=sensitive, predictions=yhat > threshold, labels=y
    )
    comparison.instance(f"Threshold {threshold:.1f}", report)
comparison = comparison.build()
comparison.maxdiff.show(env=fb.export.Html, depth=0)
# comparison.maxdiff.explain.show(env=fb.export.Html, depth=0)

#
# # comparison = fb.reduction.mean(comparison.acc.explain)
# comparison = comparison.filter(fb.reduction.mean).filter()
# dict = comparison.to_dict()
# comparison = fb.core.Value.from_dict(dict)
#
# comparison.show()
#
#
# """
# print(comparison)
#
# value = (comparison
#           |fb.reduction.minimum
#           |fb.measures.pr)
#
# print(value)
#
# for v in value.flatten():
#     print(v)
# """

# print(json.dumps((report_over_time|pr|minimum).serialize(), indent="  "))
