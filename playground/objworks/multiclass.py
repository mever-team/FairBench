import fairbench as fb
import numpy as np

x, y, yhat = fb.bench.tabular.compas(test_size=0.5, predict="probabilities")

sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
sensitive = sensitive.intersectional().strict()

yhat = np.round(yhat * 3)
y = y * 3

report = fb.reports.pairwise(multipredictions=yhat, multilabels=y, sensitive=sensitive)
report.show(env=fb.export.ConsoleTable(sideways=False))
