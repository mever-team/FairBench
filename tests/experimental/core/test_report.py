from fairbench import v2 as fb
import fairbench.v1 as fb1
import numpy as np


def test_sensitive_conversion():
    fork = fb1.Fork(men=[1, 0, 1], women=[0, 1, 0])
    sensitive = fb.Sensitive(fork.branches())
    assert len(list(sensitive.keys())) == len(fork.branches())
    for key in sensitive.keys():
        assert np.abs(sensitive[sensitive[key]] - np.array(fork[key])).sum() == 0


def test_vsany():
    x, y, yhat = fb1.bench.tabular.bank(predict="probabilities")
    sensitive = fb1.Fork(fb1.categories @ x["marital"], fb1.categories @ x["education"])
    sensitive = sensitive.intersectional().strict()

    report = fb.reports.vsall(
        sensitive=sensitive,
        predictions=yhat > 0.5,
        labels=y,
        scores=yhat,
        targets=y,
    )

    report.show(env=fb.export.Console(ansiplot=True))
    report.help()


def test_pairwise():
    x, y, yhat = fb1.bench.tabular.bank()
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


def test_investigators():
    x, y, yhat = fb1.bench.tabular.bank()
    sensitive = fb1.Fork(fb1.categories @ x["marital"], fb1.categories @ x["education"])
    sensitive = sensitive.intersectional().strict()

    fb.reports.pairwise(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
    ).filter(
        fb.investigate.DeviationsOver(0.2)
    ).filter(fb.investigate.IsBias).show()


def test_progress():
    x, y, yhat = fb1.bench.tabular.bank()

    cats = fb1.categories @ x["marital"]
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


def test_multiclass():
    x, y, yhat = fb1.bench.tabular.bank()
    sensitive = fb1.Fork(fb1.categories @ x["marital"], fb1.categories @ x["education"])
    sensitive = sensitive.intersectional().strict()
    y = fb1.categories @ y
    yhat = fb1.categories @ yhat

    report = fb.reports.pairwise(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
        scores=yhat,
        targets=y,
    )

    report.acc.show(fb.export.ConsoleTable)


def test_attachment_to_measures():
    x, y, yhat = fb1.bench.tabular.bank()
    sensitive = fb1.Fork(fb1.categories @ x["marital"], fb1.categories @ x["education"])
    sensitive = sensitive.intersectional().strict()
    y = fb1.categories @ y
    yhat = fb1.categories @ yhat

    report = fb.reports.pairwise(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
        scores=yhat,
        targets=y,
        attach_branches_to_measures=True,
    )

    report.accFalse.show(fb.export.ConsoleTable)
