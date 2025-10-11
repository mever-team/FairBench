import fairbench as fb

x, y, yhat = fb.bench.tabular.compas(test_size=0.5)

sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
sensitive = sensitive.intersectional().strict()

yhat = fb.Dimensions(fb.categories @ yhat)
y = fb.Dimensions(fb.categories @ y)

report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
report.show(env=fb.export.ConsoleTable(sideways=False))
report.filter(fb.investigate.BL(0.1)).show()


report.show(env=fb.export.HtmlBars, depth=4)
