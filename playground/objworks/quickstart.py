import fairbench as fb

test, y, yhat = fb.bench.tabular.adult()

sensitive = fb.Dimensions(
    fb.categories @ test[8], fb.categories @ test[9]
)  # analyse the gender and race columns
sensitive = sensitive.intersectional()  # automatically find non-empty intersections
sensitive = sensitive.strict()  # keep only intersections that have no children

report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
report.show(fb.export.Html(legend=False), depth=2)
