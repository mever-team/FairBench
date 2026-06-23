import fairbench as fb

x, y, yhat = fb.bench.tabular.compas(test_size=0.5)

sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
sensitive = sensitive.intersectional().strict()
#
# yhat = fb.Dimensions(fb.categories @ yhat)
# y = fb.Dimensions(fb.categories @ y)

print(sensitive.sum())

report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
report.min.acc.show(env=fb.export.Console)

report.show(env=fb.export.ConsoleTable)
report.help()
report = report.filter(fb.investigate.Stamps)
report.show(env=fb.export.Html(horizontal=True), depth=1)
# report.filter(fb.investigate.DeviationsOver(0.2)).show()


# report.filter(fb.investigate.DeviationsOver(0.2)).show(env=fb.export.HtmlBars, depth=2)
