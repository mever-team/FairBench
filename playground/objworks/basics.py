from fairbench import v2 as fb
import fairbench.v1 as fb1

x, y, yhat = fb1.bench.tabular.bank()
sensitive = fb1.Fork(fb1.categories @ x["marital"])
# sensitive = sensitive.intersectional().strict()
# y = fb1.categories @ y
# yhat = fb1.categories @ yhat

report = fb.reports.pairwise(
    sensitive=sensitive, predictions=yhat, labels=y, scores=yhat, targets=y, top=50
)

report.filter(fb.investigate.DeviationsOver(0.2)).show(fb.export.ConsoleTable)
