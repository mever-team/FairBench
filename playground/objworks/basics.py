from fairbench.experimental import v2 as fb
import fairbench as deprecated

x, yhat, y = deprecated.bench.tabular.bank()

cats = deprecated.categories @ x["marital"]
cats = {k: v.numpy() for k, v in cats.items()}

sensitive = fb.Sensitive(cats)
report = fb.report(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
    measures=[
        fb.measures.pr,
        fb.measures.tpr,
        fb.measures.tnr,
        fb.measures.tar,
        fb.measures.trr,
    ],
    reductions=[
        fb.reduction.min,
        fb.reduction.wmean,
        fb.reduction.maxrel,
        fb.reduction.maxdiff,
        fb.reduction.std,
    ],
)

#fb.export.static(report, depth=10).display()


fb.export.static(report, env=fb.export.formats.WebApp(), depth=1).display()
