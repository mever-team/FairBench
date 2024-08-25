import fairbench as fb

# testing heterogeneous setting
test, y, yhat = fb.demos.adult(predict="probabilities")
s = fb.Fork(fb.categories @ test[9])  # test[8] is a pandas column with race values

report = fb.multireport(
    scores=yhat, predictions=(yhat > 0.5), labels=y, sensitive=s, top=50
)
fb.describe(report)

# report = fb.accreport(
#    predictions=(yhat > 0.5), labels=y, sensitive=s, metrics=fb.common_adhoc_metrics
# )
# fb.describe(report)
# report2 = fb.unireport(predictions=(yhat > 0.5), labels=y, sensitive=s, top=50)
# report = fb.combine(report, report2)
fb.interactive_html(report)
