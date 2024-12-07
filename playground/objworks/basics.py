from fairbench.experimental import v2 as fb
import fairbench as deprecated

x, yhat, y = deprecated.bench.tabular.bank()
sensitive = deprecated.Fork(deprecated.categories @ x["marital"])

report = fb.reports.vsall(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
    scores=yhat,
    targets=y,
)

#fb.export.help(report)
report.min.acc.show()
#report.show(env=fb.export.WebApp())
