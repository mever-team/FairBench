import fairbench as fb

test, y, yhat = fb.bench.tabular.bank()

sensitive = fb.Dimensions(fb.categories @ test["education"])
report = fb.reports.pairwise(sensitive=sensitive, predictions=yhat, labels=y)

# report.filter(fb.investigate.DeviationsOver(0.2)).show(env=fb.export.ConsoleTable)


print(float(report.pr.maxrel))
