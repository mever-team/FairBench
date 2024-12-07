from fairbench.experimental import v2 as fb
import fairbench.v1 as fb1

x, yhat, y = fb1.bench.tabular.bank()
sensitive = fb1.Fork(
    fb1.categories @ x["marital"], fb1.categories @ x["education"]
).intersectional()

report = fb.reports.pairwise(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
    scores=yhat,
    targets=y,
)

# fb.export.help(report)
report.std.show(env=fb.export.Console(ansiplot=True))
report.std.help()
# report.show(env=fb.export.WebApp())
