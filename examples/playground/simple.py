import fairbench as fb

test, y, yhat = fb.demos.adult()
_, _, scores = fb.demos.adult(predict="probabilities")
sensitive = fb.Fork(fb.categories @ test[9])
report = fb.multireport(predictions=yhat, scores=scores, labels=y, sensitive=sensitive)

stamps = fb.combine(
    fb.stamps.prule(report),
    fb.stamps.dfpr(report),
    fb.stamps.dfnr(report),
    fb.stamps.abroca(report),
    fb.stamps.accuracy(report),
    fb.stamps.four_fifths_rule(report),
)
# print(fb.modelcards.tohtml(stamps, show=False))

fb.interactive(report, browser=True)
