import fairbench as fb

# testing heterogeneous setting
test, y, yhat = fb.demos.adult(predict="probabilities")
s = fb.Fork(fb.categories @ test[9])  # test[8] is a pandas column with race values

report = fb.multireport(
    scores=yhat, predictions=(yhat > 0.5), labels=y, sensitive=s, top=50
)
# report2 = fb.unireport(predictions=(yhat > 0.5), labels=y, sensitive=s, top=50)
# report = fb.combine(report, report2)
# fb.interactive(report)
print(report.accuracy.minratio.explain.explain)
# print(report.auc.maxbarea.explain.explain)
# fb.visualize(report.auc.maxbarea.explain.explain)
# fb.visualize(report.avgscore.maxbarea.explain.explain.curve)
