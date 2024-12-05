from fairbench.experimental import v2 as fb
import fairbench as deprecated

x, yhat, y = deprecated.bench.tabular.bank()

cats = deprecated.categories @ x["marital"]
cats = {k: v.numpy() for k, v in cats.items()}

sensitive = fb.Sensitive(cats)
report1 = fb.report(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
    measures=[
        fb.measures.pr,
        fb.measures.tpr,
        fb.measures.tnr,
        fb.measures.tar,
        fb.measures.trr,
        fb.measures.mabs,
        fb.measures.rmse,
        fb.measures.pinball,
    ],
    reductions=[
        fb.reduction.min,
        fb.reduction.wmean,
        fb.reduction.maxrel,
        fb.reduction.maxdiff,
        fb.reduction.std,
    ],
)
yhat = 1 - yhat
report2 = fb.report(
    sensitive=sensitive,
    predictions=yhat,
    labels=y,
    measures=[
        fb.measures.pr,
        fb.measures.tpr,
        fb.measures.tnr,
        fb.measures.tpr,
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

# fb.export.console(report, depth=1)


comparison = fb.Comparison("time")
comparison.instance("Day 1", report1)
comparison.instance("Day 2", report2)
comparison.instance("Day 3", report1)
comparison = comparison.build()


comparison = comparison | fb.reduction.maxrel


fb.export.static(comparison).display()
comparison = fb.reduction.mean(comparison.values("reduction measure"))

fb.help(fb.Comparison)
fb.help(fb.measures.tpr)
fb.help(comparison)
#fb.export.static(comparison, depth=1, env=fb.export.formats.WebApp()).display()

fb.export.static(comparison).display()

"""
print(comparison)

value = (comparison
          |fb.reduction.minimum
          |fb.measures.pr)

print(value)

for v in value.flatten():
    print(v)
"""

# print(json.dumps((report_over_time|pr|minimum).serialize(), indent="  "))
