import fairbench as fb
x, y, yhat= fb.bench.tabular.adult()

sensitive = fb.Fork(fb.categories@x[8])
report1 = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
report2 = report1
report3 = report1

reports = fb.Fork(t1=report1, t2=report2, t3=report3)
std = fb.reduce(reports.min, fb.std)

print(std.explain)