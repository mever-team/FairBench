import fairbench as fb

test, y, yhat = fb.demos.compas()
s = fb.Fork(fb.categories @ test["sex"])
# = fb.Fork(fb.categories @ test[9], fb.categories @ test[8]).intersectional()
report = fb.multireport(predictions=yhat, labels=y, sensitive=s)

fb.interactive(report, port=8089)
