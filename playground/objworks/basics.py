import fairbench as fb

x, y, yhat = fb.bench.tabular.compas(test_size=0.5)

sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
sensitive = sensitive.intersectional().strict()

yhat = ["yes" if val == 1 else "no" for val in yhat]
y = ["yes" if val == 1 else "no" for val in y]

yhat = fb.Dimensions(fb.categories @ yhat)
y = fb.Dimensions(fb.categories @ y)

report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
# report.filter(fb.investigate.Stamps).show(env=fb.export.Html(horizontal=True), depth=1)
report.show()
report.show(env=fb.export.ConsoleTable)
