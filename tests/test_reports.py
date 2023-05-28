import fairbench as fb
import numpy as np
from .test_forks import environment


def test_areport():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.report(predictions=predictions, labels=labels, sensitive=sensitive, metrics=[fb.accuracy, fb.pr])
        assert report.pr.men == fb.areport(predictions=predictions, labels=labels, sensitive=sensitive, metrics=fb.pr).men
        assert report.pr.women == fb.areport(predictions=predictions, labels=labels, sensitive=sensitive, metrics=fb.pr).women
        assert report.pr.nonbin == fb.areport(predictions=predictions, labels=labels, sensitive=sensitive, metrics=fb.pr).nonbin
        assert len(fb.areport(predictions=predictions, labels=labels, sensitive=sensitive, metrics=[fb.accuracy, fb.pr])) == 2
        assert len(fb.areport(predictions=predictions, labels=labels, sensitive=sensitive, metrics={"acc": fb.accuracy, "pr": fb.pr})) == 2


def test_multireport():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        report = fb.multireport(predictions=predictions, labels=labels, sensitive=sensitive)
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
        report = fb.binreport(predictions=predictions, labels=labels, sensitive=sensitive)
        assert report.men.accuracy == 1
        assert report.nonbin.prule == 0
        assert report.accuracy.men == 1
        assert report.prule.nonbin == 0