import fairbench as fb

test, y, yhat = fb.demos.adult(predict="probabilities")
s = fb.Fork(fb.categories@test[8])
report = fb.multireport(scores=yhat, labels=y, sensitive=s, top=50)

#fb.describe(report.auc)
#fb.interactive(report)
fb.visualize(report.maxbarea.arepr.explain.explain.curve, xrotation=30)
