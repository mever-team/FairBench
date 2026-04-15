from fairbench import v2 as fb
import fairbench as deprecated

x, y, yhat = deprecated.bench.tabular.bank()

sensitive = fb.Dimensions(
    {"marital": fb.categories @ x["marital"], "age": fb.fuzzy @ x["age"]}
)
print(sensitive.branches().keys())
report = fb.reports.pairwise(
    sensitive=sensitive,
    predictions={"true": yhat, "false": 1 - yhat},
    labels={"true": y, "false": 1 - y},
)

print(report.filter(fb.investigate.Stamps)) # summary print
report.filter(fb.investigate.Stamps).show(
    env=fb.export.Html, depth=2
) # show stamps as html

