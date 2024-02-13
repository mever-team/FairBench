import fairbench as fb

# testing heterogeneous setting
test, y, yhat = fb.demos.adult(predict="probabilities")
s = fb.Fork(fb.individuals @ test[8][:10])  # test[8] is a pandas column with race values
report = fb.multireport(scores=yhat[:10], targets=y[:10], sensitive=s)
fb.describe(report)

# fb.visualize(report)
fb.interactive(report)
# fb.visualize(report.maxbarea.arepr.explain.explain.curve, xrotation=30)
