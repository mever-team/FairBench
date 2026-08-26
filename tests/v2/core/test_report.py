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
    assert str(report) == report.show(fb.export.ToString)


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
        measures=[fb.measures.mcc],
        reductions=[
            fb.reduction.stdx2,
            fb.reduction.maxdiff,
            fb.reduction.gm,
            fb.reduction.std,
        ],
    )

    report.show(env=v2.export.Html(view=False, filename="temp"), depth=1)
    report.show(env=v2.export.Html(view=False, filename="temp"), depth=1)
    report.show(env=v2.export.HtmlTable(view=False, filename="temp"), depth=1)
    report.show(env=v2.export.HtmlTable(view=False, filename="temp"), depth=1)
    report.show(env=v2.export.HtmlBars(view=False, filename="temp"), depth=1)
    report.filter(v2.investigate.BL).show(
        env=v2.export.Html(view=False, filename="temp"), depth=1
    )
    report.filter(v2.investigate.Stamps).show(
        env=v2.export.Html(view=False, filename="temp"), depth=1
    )
    report.filter(v2.investigate.Stamps).show(
        env=v2.export.HtmlTable(view=False, filename="temp"), depth=1
    )
    report.filter(v2.investigate.Stamps).show(
        env=v2.export.HtmlTable(view=False, filename="temp"), depth=1
    )
    report.filter(v2.investigate.Stamps).show(
        env=v2.export.HtmlBars(view=False, filename="temp"), depth=1
    )
    report.filter(v2.investigate.Stamps).show(
        env=v2.export.Html(
            view=False, filename="temp", distributions=True, horizontal_bars=True
        ),
        depth=2,
    )
    report.maxdiff.show()  # console is the default
    report.show(v2.export.ConsoleTable)


def test_vsall():
    x, y, yhat = fb.bench.tabular.bank(predict="probabilities")
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()

    report = fb.reports.vsall(
        sensitive=sensitive.branches(),  # the most out-of-the-blue input that makes sense: a dict
        predictions=yhat > 0.5,
        labels=y,
        scores=yhat,
        targets=y,
    )

    report.show(env=fb.export.Console(ansiplot=True))
    report.help()

    report.acc.min.testeq(
        fb.quick.vsall_acc_min(
            sensitive=sensitive,
            predictions=yhat > 0.5,
            labels=y,
        ).float()
    )


def test_conflate():
    x, y, yhat = fb.bench.tabular.bank(predict="probabilities")
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()

    report = fb.reports.conflate(
        sensitive=sensitive.branches(),  # the most out-of-the-blue input that makes sense: a dict
        predictions=yhat > 0.5,
        labels=y,
        scores=yhat,
        targets=y,
    )

    report.maxrel.acc.show(env=fb.export.ConsoleTable())
    report.help()


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
        multipredictions=yhat * 2,
        multilables=y * 2,
    )
    report.min.acc.show()
    report.min.acc.help()
    report.acc.min.show()
    report.gmi.min.show()

    report.acc.min.testeq(
        fb.quick.pairwise_acc_min(
            sensitive=sensitive,
            predictions=yhat,
            labels=y,
        ).float()
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


def test_worst():
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
        .filter(fb.investigate.IsBias, fb.investigate.WorstCase)
        .show(fb.export.ToString)
    )
    assert "worst bias" in serialized


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


def test_number_of_measures():
    items = list(item for item in fb.quick)
    num = len(items)
    print(str(num) + " measures can be computed.")
    fb.quick.help()
    assert num > 400


def test_score_handling():
    y_true = np.array([10.2, 15.5, 8.1, 20.0, 12.3, 9.8, 18.4, 11.1])
    y_pred = np.array([10.0, 15.0, 9.0, 19.0, 11.0, 9.5, 25.0, 10.5])
    groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    sensitive = fb.Dimensions(fb.categories @ groups)
    for score_bound in ["unbounded", "auto", "normalized", "standardized"]:
        report = fb.reports.pairwise(
            scores=y_pred, targets=y_true, sensitive=sensitive, score_bound=score_bound
        )
        report.show(env=fb.export.Console(ansiplot=False))


def test_amplification():
    # BASE REPORT
    baseline_x, baseline_y, baseline_yhat = fb.bench.tabular.bank(test_size=0.2)
    baseline_sensitive = fb.Dimensions(
        fb.categories @ baseline_x["marital"], fb.categories @ baseline_x["education"]
    )
    baseline_sensitive = baseline_sensitive.intersectional().strict()
    baseline_report = fb.reports.pairwise(
        multipredictions=baseline_yhat,
        multilabels=baseline_y,
        sensitive=baseline_sensitive,
    )

    # AFTER "MITIGATION" (just more samples here)
    x, y, yhat = fb.bench.tabular.bank(test_size=0.5)
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()

    # INVESTIGATE AMPLIFICATION
    report = fb.reports.pairwise(
        multipredictions=yhat, multilabels=y, sensitive=sensitive
    )
    report = report.filter(fb.investigate.Amplification(base=baseline_report))
    report.show(env=fb.export.Console(ansiplot=False), depth=2)


def test_amplification_vs_base_values():
    manual_pr_values = fb.core.report_from_dims(
        fb.v1.Fork({"male": 0.3, "female": 0.4})
    )
    manual_y = [1, 0, 1, 1, 1, 0]
    manual_yhat = [1, 0, 1, 0, 1, 0]
    manual_sensitive = fb.Dimensions(
        fb.categories @ ["male", "male", "male", "female", "female", "female"]
    )
    manual_report = fb.reports.pairwise(
        labels=manual_y, predictions=manual_yhat, sensitive=manual_sensitive
    )

    print(manual_pr_values)
    print(manual_report.min.pr.details)

    manual_report.min.pr.details.filter(
        fb.investigate.Amplification(
            base=manual_pr_values, default_to_zero_targets=True
        )
    ).show(depth=2)


def test_report_deviation():
    # BASE REPORT
    baseline_x, baseline_y, baseline_yhat = fb.bench.tabular.bank(test_size=0.2)
    baseline_sensitive = fb.Dimensions(
        fb.categories @ baseline_x["marital"], fb.categories @ baseline_x["education"]
    )
    baseline_sensitive = baseline_sensitive.intersectional().strict()
    baseline_report = fb.reports.pairwise(
        multipredictions=baseline_yhat,
        multilabels=baseline_y,
        sensitive=baseline_sensitive,
    )

    # TEST WORST CASE OF BASELINE
    baseline_report.filter(fb.investigate.IsBias, fb.investigate.WorstCase).show(
        env=fb.export.Console(ansiplot=False)
    )

    # AFTER "MITIGATION" (just more samples here)
    x, y, yhat = fb.bench.tabular.bank(test_size=0.5)
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()

    # INVESTIGATE AMPLIFICATION
    report = fb.reports.pairwise(
        multipredictions=yhat, multilabels=y, sensitive=sensitive
    )
    report = report.filter(
        fb.investigate.IsBias,
        fb.investigate.Amplification(base=baseline_report, force=True),
    )
    report.show(env=fb.export.Console(ansiplot=False), depth=2)


def test_deviation_from_desired():
    x, y, yhat = fb.bench.tabular.bank(test_size=0.5)
    sensitive = fb.Dimensions(
        fb.categories @ x["marital"], fb.categories @ x["education"]
    )
    sensitive = sensitive.intersectional().strict()
    base_report = fb.core.report_from_dims(sensitive.sum() / sensitive.shape[0])
    report = fb.reports.pairwise(predictions=yhat, sensitive=sensitive).min.pr
    report.details.filter(
        fb.investigate.SetTargets(targets=base_report), fb.investigate.WorstCase
    ).show()
    report.details.filter(
        fb.investigate.SetTargets(targets=base_report), fb.investigate.SkewIndex
    ).show()


def test_help():
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
    progress = fb.Progress("test progress")
    progress["instance"] = report
    fb.export.help(report)
    fb.export.help(progress)
    fb.export.help(fb.measures.gmi)
