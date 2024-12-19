import os

os.environ["FBINTERACTIVE"] = "True"


import fairbench as fb

x, y, yhat = fb.bench.tabular.compas(test_size=0.5)
print(x)


sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
print(sensitive)

sensitive = sensitive.intersectional()
print(sensitive)

report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)

report.filter(fb.investigate.Stamps).show(env=fb.export.Html, depth=1)

report.help()
