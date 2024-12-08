from fairbench import v2 as fb
import fairbench.v1 as fb1

x, yhat, y = fb1.bench.tabular.bank()
sensitive = fb1.Fork(fb1.categories @ x["marital"], fb1.categories @ x["education"])
sensitive = sensitive.intersectional().strict()
y = fb1.categories @ y
yhat = fb1.categories @ yhat

report = fb.reports.pairwise(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
)

report.show(env=fb.export.Console(ansiplot=True))
