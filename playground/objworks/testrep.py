import fairbench as fb

x, y, yhat = fb.bench.tabular.bank(predict="probabilities")
sensitive = fb.Dimensions(fb.categories @ x["marital"], fb.categories @ x["education"])
sensitive = sensitive.intersectional().strict()

report = fb.reports.vsall(
    sensitive=sensitive,
    predictions=yhat > 0.5,
    labels=y,
    scores=yhat,
    targets=y,
)

report.show(env=fb.export.Console(ansiplot=True))
report.help()
