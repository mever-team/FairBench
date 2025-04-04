from fairbench import v2 as fb
import fairbench as deprecated

x, y, yhat = deprecated.bench.tabular.bank()

cats = deprecated.categories @ x["marital"]
cats = {k: v.numpy() for k, v in cats.items()}

sensitive = fb.Sensitive(cats)
report1 = fb.reports.pairwise(sensitive=sensitive, predictions=yhat, labels=y)

comparison = fb.Progress("variations")
comparison.instance("Take 1", report1)
comparison.instance("Take 2", report1)
comparison = comparison.build()

comparison.details.show()
