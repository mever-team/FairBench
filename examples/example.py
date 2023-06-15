import fairbench as fb

test, y, yhat = fb.demos.bank()
s = fb.Fork(fb.categories@test['marital'])
report = fb.multireport(predictions=yhat, labels=y, sensitive=s)
print(report.explain)
