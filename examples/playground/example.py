import fairbench as fb

test, y, yhat = fb.demos.adult()
s = fb.Fork(fb.categories @ test[9])
report = fb.multireport(predictions=yhat, labels=y, sensitive=s)

# fb.interactive(report)

stamp = fb.combine(
    fb.stamps.prule(report),
    fb.stamps.accuracy(report),
    fb.stamps.four_fifths_rule(report),
)
fb.modelcards.tohtml(stamp, show=True)
