from fairbench.experimental import v2 as fb
import fairbench as deprecated

x, yhat, y = deprecated.bench.tabular.bank()
sensitive = deprecated.Fork(deprecated.categories @ x["marital"], deprecated.categories @ x["education"]).intersectional()

report = fb.reports.pairwise(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
    scores=yhat,
    targets=y,
)

#fb.export.help(report)
report.std.show(env=fb.export.Console(ansiplot=True))
report.std.help()
#report.show(env=fb.export.WebApp())
