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

# report.min.roc.show()
report.filter(fb.investigate.Threshold(0.2)).show(fb.export.ConsoleTable)
# report.filter(fb.investigate.MostProblematic).show(fb.export.ConsoleTable)
# report.filter(fb.investigate.Caveats).show(fb.export.ConsoleTable)
# (fb.export.ConsoleTable)
# fb.export.Html

# fb.reduction.min(report.min.explain).show()
