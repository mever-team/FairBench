import fairbench as fb

test, y, yhat = fb.demos.adult()
s = fb.Fork(fb.categories @ test[9])
report = fb.multireport(predictions=yhat, labels=y, sensitive=s)

#fb.interactive(report)

stamp = fb.combine(
    fb.stamps.prule(report),
    fb.stamps.accuracy(report),
    #fb.stamps.three_fourths(report),
    fb.stamps.eighty_rule(report)
)
print(fb.modelcards.tomarkdown(stamp))

# fb.interactive(report)
