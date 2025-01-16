import fairbench as fb

yhat = [0]*10
y = [0]*8+[1,1]
sensitive = fb.Fork(fb.categories@[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # see why weighted mean doesn't work

report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
fb.text_visualize(report.accuracy)

"""import fairbench as fb
x, y, yhat= fb.bench.tabular.adult()

sensitive = fb.Fork(fb.categories@x[8])
report1 = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
report2 = report1
report3 = report1

reports = fb.Fork(t1=report1, t2=report2, t3=report3)
std = fb.reduce(reports.min, fb.std)

print(std.explain)"""