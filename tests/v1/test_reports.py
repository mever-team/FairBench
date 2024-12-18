from fairbench import v1 as fb
import numpy as np
from .test_forks import environment


def test_report_onemetric():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.multireport(
            predictions=predictions,
            labels=labels,
            sensitive=sensitive,
            metrics=fb.accuracy,
        )
        assert report.min.accuracy.value == 0


def test_areport():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.report(
            predictions=predictions,
            labels=labels,
            sensitive=sensitive,
            metrics=[fb.accuracy, fb.pr],
        )
        assert (
            report.pr.men
            == fb.areport(
                predictions=predictions,
                labels=labels,
                sensitive=sensitive,
                metrics=fb.pr,
            ).men
        )
        assert (
            report.pr.women
            == fb.areport(
                predictions=predictions,
                labels=labels,
                sensitive=sensitive,
                metrics=fb.pr,
            ).women
        )
        assert (
            report.pr.nonbin
            == fb.areport(
                predictions=predictions,
                labels=labels,
                sensitive=sensitive,
                metrics=fb.pr,
            ).nonbin
        )
        assert (
            len(
                fb.areport(
                    predictions=predictions,
                    labels=labels,
                    sensitive=sensitive,
                    metrics=[fb.accuracy, fb.pr],
                )
            )
            == 2
        )
        assert (
            len(
                fb.areport(
                    predictions=predictions,
                    labels=labels,
                    sensitive=sensitive,
                    metrics={"acc": fb.accuracy, "pr": fb.pr},
                )
            )
            == 2
        )


def test_multireport():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.multireport(
            predictions=predictions, labels=labels, sensitive=sensitive
        )
        assert report.min.accuracy.value == 0
        assert report.min.accuracy.explain.men == 1
        assert report.minratio.pr.value == 0


def test_binreport():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.binreport(
            predictions=predictions, labels=labels, sensitive=sensitive
        )
        assert report.men.accuracy == 1
        assert report.nonbin.prule == 0
        assert report.accuracy.men == 1
        assert report.prule.nonbin == 0


def test_biasreport():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.biasreport(
            predictions=predictions, labels=labels, sensitive=sensitive
        )
        assert report.notone.accuracy.value == 1
        assert report.gini.accuracy.explain.men == 1
        assert report.maxrdiff.pr.value == 1
        assert report.maxdiff.pr.value < 0.7


def test_fuzzyreport():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.fuzzyreport(
            predictions=predictions, labels=labels, sensitive=sensitive
        )
        assert abs(report.notone.accuracy - 1) < 1.0e-6
        assert abs(report["tprodrdiff[vsAny]"].accuracy - 1) < 1.0e-6
        assert abs(report["tprodrdiff[vsAny]"].pr - 1) < 1.0e-6
        assert abs(report["tprodrdiff[vsAny]"].tpr - 1) < 1.0e-6
        assert abs(report["tprodrdiff[vsAny]"].tnr - 1) < 1.0e-6
        assert report["tlukadiff[vsAny]"].pr < 0.9


def test_accreport():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.accreport(
            predictions=predictions, labels=labels, sensitive=sensitive
        )
        assert report.men.accuracy == 1
        assert report.nonbin.pr == 0


def test_unireport():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.unireport(
            predictions=predictions, labels=labels, sensitive=sensitive
        )
        assert report.branches()["min"].accuracy.value == 0
        assert report.branches()["min"].accuracy.explain.men == 1
        assert report.branches()["minratio[vsAny]"].pr.value == 0


def test_report_combination():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report1 = fb.binreport(
            predictions=predictions, labels=labels, sensitive=sensitive
        )
        report2 = fb.multireport(
            predictions=predictions, labels=labels, sensitive=sensitive
        )
        report = fb.combine(report1, report2)
        assert "min" in report.branches()
        assert "women" in report.branches()


def test_extract():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.multireport(
            predictions=predictions, labels=labels, sensitive=sensitive
        )
        extracted = fb.Fork(acc=report.min.accuracy, prule=report.pr.minratio)
        assert report.min.accuracy == extracted.acc
        assert report.minratio.pr == extracted.prule


def test_extract_comparison():
    # for _ in environment():  # TODO: fix extract for distributed environment
    men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
    predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    report1 = fb.multireport(
        predictions=predictions, labels=labels, sensitive=sensitive
    )
    report2 = fb.multireport(
        predictions=predictions,
        labels=labels,
        sensitive=fb.Fork(men=men, women=1 - men),
    )
    report = fb.Fork(report1=report1, report2=report2)
    extracted = fb.extract(acc=report.min.accuracy, prule=report.pr.minratio)
    assert report1.min.accuracy.value == extracted.acc.report1.value
    assert report2.minratio.pr.value == extracted.prule.report2.value


def test_disparity_metrics():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        report = fb.report(
            predictions=predictions,
            sensitive=sensitive,
            metrics=[fb.cvdisparity, fb.eqrep],
        )
        assert report.men.eqrep <= 1
        assert report.men.cvdisparity >= 0


def test_rates():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        report = fb.report(
            predictions=predictions,
            labels=labels,
            sensitive=sensitive,
            metrics=[fb.tpr, fb.tnr, fb.fpr, fb.fnr],
        )

        assert report.men.tpr == 1
        assert report.men.tnr == 1
        assert report.men.fpr == 0
        assert report.men.fnr == 0
