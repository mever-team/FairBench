from fairbench.v2 import v2 as fb
import fairbench.v1 as fb1
import numpy as np


def test_sensitive_conversion():
    fork = fb1.Fork(men=[1, 0, 1], women=[0, 1, 0])
    sensitive = fb.Sensitive(fork.branches())
    assert len(list(sensitive.keys())) == len(fork.branches())
    for key in sensitive.keys():
        assert np.abs(sensitive[sensitive[key]] - np.array(fork[key])).sum() == 0


def test_vsany():
    x, yhat, y = fb1.bench.tabular.bank()
    sensitive = fb1.Fork(fb1.categories @ x["marital"], fb1.categories @ x["education"])
    sensitive = sensitive.intersectional().strict()

    report = fb.reports.vsall(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
        scores=yhat,
        targets=y,
    )

    report.show(env=fb.export.Console(ansiplot=True))
    report.help()


def test_pairwise():
    x, yhat, y = fb1.bench.tabular.bank()
    sensitive = fb1.Fork(fb1.categories @ x["marital"], fb1.categories @ x["education"])
    sensitive = sensitive.intersectional().strict()

    report = fb.reports.pairwise(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
    )
    report.min.acc.show()
    report.min.acc.help()
    report.acc.min.show()


def test_progress():
    from fairbench.v2 import v2 as fb
    import fairbench as deprecated

    x, yhat, y = deprecated.bench.tabular.bank()

    cats = deprecated.categories @ x["marital"]
    cats = {k: v.numpy() for k, v in cats.items()}

    sensitive = fb.Sensitive(cats)
    report1 = fb.reports.pairwise(sensitive=sensitive, predictions=yhat, labels=y)
    yhat = 1 - yhat
    report2 = fb.reports.pairwise(sensitive=sensitive, predictions=yhat, labels=y)

    comparison = fb.Progress("time")
    comparison.instance("Day 1", report1)
    comparison.instance("Day 2", report2)
    comparison.instance("Day 3", report1)

    assert fb.Progress(comparison.status).status.exists()

    comparison = comparison.build()
    comparison = fb.core.Value.from_dict(comparison.to_dict())  # hard test

    comparison = fb.reduction.mean(comparison.min.explain)

    comparison = fb.core.Value.from_dict(comparison.to_dict())  # hard test
    comparison.details.show()
