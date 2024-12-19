import os

os.environ["FBINTERACTIVE"] = "True"
import fairbench as fb


x, y, yhat = fb.bench.tabular.compas(test_size=0.5)
print(x)


sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
print(sensitive)


report = fb.reports.pairwise(
    predictions=fb.categories @ yhat, labels=fb.categories @ y, sensitive=sensitive
)
report.show(fb.export.HtmlTable())
