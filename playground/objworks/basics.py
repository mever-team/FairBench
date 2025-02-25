import fairbench as fb

x, y, yhat = fb.bench.tabular.compas(test_size=0.5)

sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
sensitive = sensitive.intersectional()
report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)

report.filter(fb.investigate.DeviationsOver(0.2)).show(env=fb.export.ConsoleTable)
report.maxrel.pr.show(env=fb.export.Console)
