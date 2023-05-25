import fairbench as fb
import numpy as np
from .test_forks import environment


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
