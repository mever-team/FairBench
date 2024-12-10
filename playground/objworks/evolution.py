from fairbench import v2 as fb
import fairbench as deprecated

x, y, yhat = deprecated.bench.tabular.bank()

cats = deprecated.categories @ x["marital"]
cats = {k: v.numpy() for k, v in cats.items()}

sensitive = fb.Sensitive(cats)
report1 = fb.reports.pairwise(sensitive=sensitive, predictions=yhat, labels=y)
yhat = 1 - yhat
report2 = fb.reports.pairwise(sensitive=sensitive, predictions=yhat, labels=y)

comparison = fb.Progress("time")
comparison.instance("Day 1", report1)
comparison.instance("Day 2", report2)
comparison.instance("Day 3", report1)
comparison.instance("Day 4", report1)
comparison.instance("Day 5", report2)
comparison.instance("Day 6", report1)
comparison = comparison.build()


# comparison.show(depth=10)

comparison = fb.reduction.mean(comparison.acc.explain)

dict = comparison.to_dict()
comparison = fb.core.Value.from_dict(dict)

comparison.details.show(fb.export.Console(ansiplot=True))


"""
print(comparison)

value = (comparison
          |fb.reduction.minimum
          |fb.measures.pr)

print(value)

for v in value.flatten():
    print(v)
"""

# print(json.dumps((report_over_time|pr|minimum).serialize(), indent="  "))
