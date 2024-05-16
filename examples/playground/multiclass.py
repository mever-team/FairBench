import fairbench as fb

test, y, yhat = fb.demos.adult()
s = fb.Fork(fb.categories @ test[9], fb.categories @ test[8]).intersectional()
# s = fb.Fork(fb.categories @ test[9])

yhat_classes = fb.categories @ yhat
y_classes = fb.categories @ y

reports = {}
for cl in y_classes:
    y = y_classes[cl]
    yhat = yhat_classes[cl]
    reports[cl] = fb.multireport(predictions=yhat, labels=y, sensitive=s)

complex_report = fb.Fork(reports)
fb.visualize(complex_report["True"].minratio.accuracy.explain, xrotation=90)

# fb.interactive(report, port=8089)
