import fairbench as fb

x, y, yhat = fb.bench.tabular.compas(test_size=0.5, predict="probabilities")
sensitive = fb.Dimensions(fb.categories @ x["race"])

# more than 300 measures dynamically generated from their name
abroca = fb.quick.pairwise_maxbarea_auc(scores=yhat, labels=y, sensitive=sensitive)
print(abroca.float())
abroca.roc.show()
