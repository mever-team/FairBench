import fairbench as fb

# testing heterogeneous setting
test, y, yhat = fb.demos.adult(predict="probabilities")
s = fb.Fork(fb.categories @ test[8])  # test[8] is a pandas column with race values
print(s)
report = fb.multireport(
    predictions=(yhat > 0.5), scores=yhat, labels=y, sensitive=s, top=50
)
report2 = fb.unireport(scores=yhat, labels=y, sensitive=s, top=50)
report = fb.combine(report, report2)
fb.describe(report)

# fb.visualize(report)
fb.interactive(report)
# fb.visualize(report.maxbarea.arepr.explain.explain.curve, xrotation=30)
