from fairbench import v2 as fb
import fairbench.v1 as fb1

x, y, yhat = fb1.bench.tabular.bank()
sensitive = fb1.Fork(fb1.categories @ x["marital"])

report = fb.reports.vsall(sensitive=sensitive, predictions=yhat, labels=y)

report.filter([fb.investigate.Stamps]).show(env=fb.export.Html, depth=1)
print(report.show(fb.export.ToJson(indent=2)))
