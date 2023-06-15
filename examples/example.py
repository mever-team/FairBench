import fairbench as fb

test, y, yhat = fb.demos.adult()
s = fb.Fork(fb.categories@test[8])
report = fb.multireport(predictions=yhat, labels=y, sensitive=s)

fb.visualize(report)
fb.visualize(report.min.explain)
