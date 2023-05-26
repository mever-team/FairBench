import fairbench as fb
import numpy as np
from .test_forks import environment


def test_reduce():
    for _ in environment():
        predictions = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        labels = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        report = fb.binreport(sensitive=sensitive, predictions=predictions, labels=labels)
        reduction = fb.reduce(report, reducer=fb.min, expand=fb.ratio)
        assert reduction.minratio.accuracy.value == 0.6666666666666667
        assert reduction.minratio.dfnr.value == -2
        assert fb.areduce(report.accuracy, reducer=fb.min, expand=fb.ratio).value == 0.6666666666666667
        assert fb.areduce(report.accuracy, reducer=fb.max, expand=fb.diff).value == 0.33333333333333326
        assert fb.areduce(report.accuracy, reducer=fb.mean, expand=fb.diff).value == 0.1481481481481481
        assert fb.areduce(report.accuracy, reducer=fb.budget, expand=fb.diff).value == -1.09861228866811
