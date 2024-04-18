import fairbench as fb
test, y, yhat = fb.demos.adult()
sensitive=fb.Fork(fb.categories@test[9], fb.categories@test[8])
report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)

report = fb.simplify(report)
print(report)
#fb.interactive(report)
