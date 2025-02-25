import fairbench as fb

x, y, yhat = fb.bench.tabular.compas(test_size=0.5)

sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
sensitive = sensitive.intersectional()
report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)

direct_values = {k: float(v) for k, v in report.pr.maxdiff.depends.items()}
print(direct_values)

ret = report.to_dict()["depends"][6]["depends"][1]
print(report.maxdiff.pr | float)
