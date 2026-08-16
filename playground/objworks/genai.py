import fairbench as fb

x, y, yhat = fb.bench.tabular.compas(test_size=0.2)
sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
sensitive = sensitive.intersectional().strict()
baseline = fb.reports.pairwise(
    multipredictions=yhat, multilabels=y, sensitive=sensitive
)

x, y, yhat = fb.bench.tabular.compas(test_size=0.5)
sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
sensitive = sensitive.intersectional().strict()
print(sensitive.sum())
report = fb.reports.pairwise(multipredictions=yhat, multilabels=y, sensitive=sensitive)

report = report.filter(fb.investigate.BiasAmplification(base=baseline))
report.show(fb.export.Html, depth=2)
