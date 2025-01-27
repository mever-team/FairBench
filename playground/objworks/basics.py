# import os
# os.environ["FBINTERACTIVE"] = "True"
import fairbench as fb

x, y, yhat = fb.bench.tabular.compas(test_size=0.5)

sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
# print(sensitive)
# sensitive = sensitive.intersectional().strict()
# print("--------------")
# print(sensitive)

report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
# print(float(report.min.acc))
# report.show(env=fb.export.WebApp)


report.show(env=fb.export.HtmlTable, depth=1)
