import fairbench as fb

test, y, yhat = fb.bench.tabular.bank()

sensitive = fb.Dimensions(fb.categories @ test["education"])
report = fb.reports.pairwise(sensitive=sensitive, predictions=yhat, labels=y)

report = report.filter(fb.investigate.BL(encounter=0.3, prune=False))
report.show(env=fb.export.Html(horizontal=True), depth=2)
