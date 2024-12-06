from fairbench.experimental import v2 as fb
import fairbench as deprecated

x, yhat, y = deprecated.bench.tabular.bank()
sensitive = deprecated.Fork(deprecated.categories @ x["marital"])

report = fb.reports.pairwise(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
    scores=yhat,
    targets=y,
)

report.show()
# fb.export.static(report, depth=10).display()
#report.show(fb.export.WebApp())
#print(report.keys())
#fb.export.static(report.samples).display()
