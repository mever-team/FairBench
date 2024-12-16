from fairbench import v2 as fb
import fairbench.v1 as fb1

x, y, yhat = fb1.bench.tabular.bank(predict="probabilities")
sensitive = fb1.Fork(fb1.categories @ x["marital"])

report = fb.reports.vsall(sensitive=sensitive, scores=yhat, labels=y)
# report.filter(fb.investigate.IsBias, fb.investigate.Stamps).show(depth=1)
report.filter(fb.investigate.Stamps).show(fb.export.Html, depth=1)
