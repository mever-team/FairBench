import fairbench as fb

test, y, yhat = fb.demos.adult(predict="probabilities")
s = fb.Fork(fb.categories @ test[9])
report = fb.unireport(scores=yhat, labels=y, sensitive=s)

text = fb.describe(report, show=False, separator=" & ", newline="\\\\\n")
print(text)

# fb.visualize(report.avgscore.maxbarea.explain.explain.curve)

# fb.describe(report)
