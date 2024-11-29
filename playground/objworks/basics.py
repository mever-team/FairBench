from fairbench import v2 as fb
import fairbench as deprecated

x, yhat, y = deprecated.bench.tabular.bank()

cats = deprecated.categories @ x["marital"]
cats = {k: v.numpy() for k, v in cats.items()}

sensitive = fb.Sensitive(cats)
report = fb.report(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
    measures=[fb.measures.pr, fb.measures.tpr, fb.measures.tnr],
    reductions=[fb.reduction.min, fb.reduction.maxrelative, fb.reduction.wmean],
)

fb.export.help(report.pr.single)
fb.export.console(report.pr.single)
