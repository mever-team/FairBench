import fairbench as fb


x, y, yhat = fb.bench.tabular.compas(test_size=0.5, predict="probabilities")

sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
# sensitive = sensitive.intersectional(min_size=10).strict()

value = fb.quick.pairwise_maxbarea_auc(scores=yhat, labels=y, sensitive=sensitive)
print(float(value))
print(value)

# value.filter(fb.investigate.DeviationsOver(0.1, prune=False)).show(env=fb.export.Html)


report = fb.reports.pairwise(
    sensitive=sensitive, scores=yhat, labels=y, predictions=yhat > 0.5, top=10
)
report.show(env=fb.export.ConsoleTable)
