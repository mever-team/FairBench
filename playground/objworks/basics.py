from fairbench import v2 as fb
import fairbench.v1 as fb1

x, y, yhat = fb1.bench.tabular.bank()
sensitive = fb1.Fork(fb1.categories @ x["marital"], fb1.categories @ x["education"])
# sensitive = sensitive.intersectional().strict()
# y = fb1.categories @ y
# yhat = fb1.categories @ yhat

report = fb.reports.pairwise(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
    scores=yhat,
    targets=y,
)

# report.min.roc.show()
report.show(env=fb.export.ConsoleTable)
# (fb.export.ConsoleTable)
# fb.export.Html

# fb.reduction.min(report.min.explain).show()
