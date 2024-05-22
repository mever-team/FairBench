import fairbench as fb

test, y, yhat = fb.demos.adult(predict="probabilities")
s = fb.Fork(fb.categories @ test[9])
# s = fb.Fork(fb.categories @ test[9], fb.categories @ test[8]).intersectional()
report = fb.combine(
    fb.unireport(scores=yhat, labels=y, sensitive=s),
    fb.multireport(scores=yhat, labels=y, sensitive=s),
)

fb.describe(report)

fb.visualize(report.avgscore.maxbarea.explain.explain.curve)

# fb.describe(report)
