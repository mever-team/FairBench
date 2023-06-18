import fairbench as fb

test, y, yhat = fb.demos.adult()#predict="probabilities")
s = fb.Fork(fb.categories@test[9])
report = fb.multireport(predictions=yhat, labels=y, sensitive=s)
report = fb.Fork(algorithm1=report, algorithms2=report, algorithms3=report)

#print(report.min.auc.explain.explain)

fb.interactive(report)

#fb.visualize(report.min.auc.explain.explain)
#fb.visualize(report.wmean.explain)
