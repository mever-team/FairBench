import fairbench as fb


test, y, scores = fb.bench.tabular.adult(predict="probabilities")
yhat = scores > 0.5
sensitive = fb.Fork(fb.categories @ test[8], fb.categories @ test[9])
# sensitive = sensitive.intersectional()

report = fb.multireport(
    predictions=yhat, labels=y, scores=scores, sensitive=sensitive, top=20
)
fb.text_visualize(report.min.auc.explain.explain, save="plot")
fb.text_visualize(report.min.avgrepr.explain.explain, save="plot")

report = fb.fuzzyreport(predictions=yhat, labels=y, scores=scores, sensitive=sensitive)
fb.text_visualize(report)

# fb.visualize(report.accuracy)
# fb.interactive(report)


"""
import fairbench as fb
report = fb.accreport( # just print performance metrics
    predictions=[1, 0, 1, 0, 0],
    labels=[1, 0, 0, 1, 0],
    metrics=[fb.accuracy, fb.pr, fb.fpr, fb.fnr])
print(report)
"""
