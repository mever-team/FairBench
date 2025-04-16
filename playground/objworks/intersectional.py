import fairbench as fb

x, y, yhat = fb.bench.tabular.bank()
print(x.keys())
sensitive = fb.Dimensions(
    marital=fb.categories @ x["marital"], education=fb.categories @ x["education"]
)
sensitive = sensitive.intersectional(min_size=100).strict()
print(sensitive)
