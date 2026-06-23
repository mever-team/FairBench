import fairbench as fb
x, y, yhat = fb.bench.tabular.bank()
sensitive = fb.Dimensions(marital=fb.categories @ x["marital"], education=fb.categories @ x["education"])
sensitive = sensitive.intersectional(min_size=100).strict()
from ansiprint import AnsiTee
with AnsiTee.activate("ansi.html"):
    print(sensitive)  # only a few large subgroups are retained