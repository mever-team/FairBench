import fairbench as fb

test, y, yhat = fb.demos.adult(predict="probabilities")
s = fb.Fork(fb.categories@test[8])
report = fb.multireport(scores=yhat, labels=y, sensitive=s)

print(report.min.auc.explain.explain)
fb.visualize(report.min.auc.explain.explain)

