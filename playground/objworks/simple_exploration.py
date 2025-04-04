import fairbench as fb

test, y, yhat = fb.bench.tabular.bank()

sensitive = fb.Dimensions(fb.categories @ test["education"])
report = fb.reports.pairwise(sensitive=sensitive, predictions=yhat, labels=y)

report.show(env=fb.export.Html(distributions=True, horizontal=True))

report.show()

print(float(report.pr.maxrel))
