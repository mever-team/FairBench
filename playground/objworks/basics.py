from fairbench import v2 as fb
import fairbench.v1 as fb1

x, y, yhat = fb1.bench.tabular.bank()
sensitive = fb1.Fork(fb1.categories @ x["marital"])

report = fb.reports.vsall(sensitive=sensitive, predictions=yhat, labels=y)

report.filter([fb.investigate.Stamps]).show(depth=1)
