from fairbench.experimental import v2 as fb
import fairbench.v1 as fb1

x, yhat, y = fb1.bench.tabular.bank()
sensitive = fb1.Fork(fb1.categories @ x["marital"], fb1.categories @ x["education"])
sensitive = sensitive.intersectional().strict()

report = fb.reports.vsall(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
    scores=yhat,
    targets=y,
)

report.min.acc.show(env=fb.export.Console(ansiplot=True))
