import fairbench as fb
import numpy as np


def test_sensitive_conversion():
    fork = fb.Dimensions(men=[1, 0, 1], women=[0, 1, 0])
    sensitive = fb.Sensitive(fork.branches())
    assert len(list(sensitive.keys())) == len(fork.branches())
    for key in sensitive.keys():
        assert np.abs(sensitive[sensitive[key]] - np.array(fork[key])).sum() == 0


def test_env():
    x, y, yhat = fb.bench.tabular.bank(predict="probabilities")
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()

    report = fb.reports.vsall(
        sensitive=sensitive,
        predictions=yhat > 0.5,
        labels=y,
        scores=yhat,
        targets=y,
    )

    assert str(report.to_dict()) == str(report.show(fb.export.ToDict))


def test_simple_report():
    from fairbench import v2
    import fairbench as fb

    sensitive = ["M", "F", "M", "F", "M", "F", "M"]
    y = [1, 1, 0, 0, 1, 0, 1]
    yhat = [1, 1, 1, 0, 0, 0, 0]

    report = fb.reports.pairwise(
        predictions=yhat,
        labels=y,
        sensitive=fb.Dimensions(fb.categories @ sensitive),
    )

    report.filter(v2.investigate.Stamps).show(
        env=v2.export.Html(view=False, filename="temp"), depth=1
    )
    report.filter(v2.investigate.Stamps).show(
        env=v2.export.Html(
            view=False, filename="temp", distributions=True, horizontal_bars=True
        ),
        depth=2,
    )
    report.maxdiff.show()  # console is the default
    report.show(v2.export.ConsoleTable)


def test_vsany():
    x, y, yhat = fb.bench.tabular.bank(predict="probabilities")
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
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

    assert report.acc.min == fb.quick.vsall_acc_min(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
    )


def test_pairwise():
    x, y, yhat = fb.bench.tabular.bank()
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()

    report = fb.reports.pairwise(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
    )
    report.min.acc.show()
    report.min.acc.help()
    report.acc.min.show()

    assert report.acc.min == fb.quick.pairwise_acc_min(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
    )


def test_exceedingly_bad_recommendation():
    x, y, yhat = fb.bench.tabular.bank()
    sensitive = fb.Dimensions(fb.fuzzy @ x["age"], fb.categories @ x["education"])
    sensitive = sensitive.intersectional().strict()

    report = fb.reports.pairwise(
        sensitive=sensitive,
        scores=yhat,
        labels=y,
    )
    report.show()


def test_investigators():
    x, y, yhat = fb.bench.tabular.bank()
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()

    fb.reports.pairwise(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
    ).filter(
        fb.investigate.DeviationsOver(0.2)
    ).filter(fb.investigate.IsBias).show()


def test_stamp_investigation():
    x, y, yhat = fb.bench.tabular.bank()
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()

    serialized = (
        fb.reports.pairwise(
            sensitive=sensitive,
            predictions=yhat,
            labels=y,
        )
        .filter(fb.investigate.Stamps)
        .show(fb.export.ToJson)
    )
    assert "worst accuracy" in serialized


def test_progress():
    x, y, yhat = fb.bench.tabular.bank()

    cats = fb.categories @ x["marital"]
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
    x, y, yhat = fb.bench.tabular.bank()
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()
    y = fb.categories @ y
    yhat = fb.categories @ yhat

    report = fb.reports.pairwise(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
        scores=yhat,
        targets=y,
    )

    report.acc.show(fb.export.ConsoleTable)


def test_attachment_to_measures():
    x, y, yhat = fb.bench.tabular.bank()
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()
    y = fb.categories @ y
    yhat = fb.categories @ yhat

    report = fb.reports.pairwise(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
        scores=yhat,
        targets=y,
        attach_branches_to_measures=True,
    )

    report.accFalse.show(fb.export.ConsoleTable)
