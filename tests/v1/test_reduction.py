from fairbench import v1 as fb
import numpy as np
from .test_forks import environment


def produce_report():
    predictions = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    labels = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
    report = fb.binreport(sensitive=sensitive, predictions=predictions, labels=labels)
    return report


def test_reduce():
    for _ in environment():
        print(produce_report())
        reduction = fb.reduce(produce_report(), reducer=fb.min, expand=fb.ratio)
        assert abs(reduction.minratio.accuracy.value - 0.666666) < 1.0e-6
        reduction = fb.reduce(produce_report(), reducer=fb.max, expand=fb.diff)
        assert reduction.maxdiff.dfnr.value == 0


def test_areduce():
    report = produce_report()
    assert (
        abs(
            fb.areduce(report.accuracy, reducer=fb.min, expand=fb.ratio)
            - 0.6666666666666667
        )
        < 1.0e-6
    )
    assert (
        abs(
            fb.areduce(report.accuracy, reducer=fb.max, expand=fb.diff)
            - 0.33333333333333326
        )
        < 1.0e-6
    )
    assert (
        abs(
            fb.areduce(report.accuracy, reducer=fb.mean, expand=fb.diff)
            - 0.1481481481481481
        )
        < 1.0e-6
    )
    assert (
        abs(
            fb.areduce(report.accuracy, reducer=fb.budget, expand=fb.diff)
            + 1.09861228866811
        )
        < 1.0e-6
    )
    assert (
        abs(
            fb.areduce(report.accuracy, reducer=fb.min, expand=fb.ratio).value
            - 0.6666666666666667
        )
        < 1.0e-6
    )
    assert (
        abs(
            fb.areduce(report.accuracy, reducer=fb.max, expand=fb.diff).value
            - 0.33333333333333326
        )
        < 1.0e-6
    )
    assert (
        abs(
            fb.areduce(report.accuracy, reducer=fb.mean, expand=fb.diff).value
            - 0.1481481481481481
        )
        < 1.0e-6
    )
    assert (
        abs(
            fb.areduce(report.accuracy, reducer=fb.budget, expand=fb.diff).value
            - -1.09861228866811
        )
        < 1.0e-6
    )
