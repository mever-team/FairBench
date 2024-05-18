import fairbench as fb

test, y, yhat = fb.demos.adult()
_, _, scores = fb.demos.adult(predict="probabilities")
sensitive = fb.Fork(fb.categories @ test[9])
report = fb.multireport(predictions=yhat, scores=scores, labels=y, sensitive=sensitive)

stamps = fb.combine(
    fb.stamps.four_fifths(report),
    fb.stamps.prule(report),
    fb.stamps.rbroca(report),
)
fb.modelcards.tohtml(stamps, show=True)

# fb.interactive(report, browser=True)
