from fairbench import v1 as fb
from .test_forks import environment


def test_accuracy():
    for _ in environment():
        assert fb.accuracy(fb.astensor([1, 1, 0, 0]), fb.astensor([1, 1, 1, 0])) < 1
        assert (
            fb.accuracy(
                fb.astensor([1, 1, 0, 0]),
                fb.astensor([1, 1, 1, 0]),
                fb.astensor([1, 1, 0, 1]),
            )
            == 1
        )


def test_tpr():
    for _ in environment():
        assert fb.tpr(fb.astensor([1, 1, 0, 0]), fb.astensor([1, 1, 1, 0])) < 1
        assert fb.tpr(fb.astensor([1, 1, 1, 0]), fb.astensor([1, 1, 0, 0])) == 1
        assert (
            fb.tpr(
                fb.astensor([1, 1, 0, 0]),
                fb.astensor([1, 1, 1, 0]),
                fb.astensor([1, 1, 0, 1]),
            )
            == 1
        )


def test_tnr():
    for _ in environment():
        assert fb.tnr(fb.astensor([1, 1, 0, 0]), fb.astensor([1, 1, 1, 0])) == 1
        assert fb.tnr(fb.astensor([1, 1, 0, 0]), fb.astensor([1, 0, 0, 0])) < 1
        assert (
            fb.tnr(
                fb.astensor([1, 1, 0, 0]),
                fb.astensor([1, 1, 1, 0]),
                fb.astensor([1, 1, 0, 1]),
            )
            == 1
        )


def test_fpr():
    for _ in environment():
        assert fb.fpr(fb.astensor([1, 1, 0, 0]), fb.astensor([1, 1, 1, 0])) == 0
        assert fb.fpr(fb.astensor([1, 1, 1, 0]), fb.astensor([1, 1, 0, 0])) > 0
        assert (
            fb.fpr(
                fb.astensor([1, 1, 1, 0]),
                fb.astensor([1, 1, 0, 0]),
                fb.astensor([1, 1, 0, 1]),
            )
            == 0
        )


def test_fnr():
    for _ in environment():
        assert fb.fnr(fb.astensor([1, 1, 0, 0]), fb.astensor([1, 1, 1, 0])) > 0
        assert fb.fnr(fb.astensor([1, 1, 0, 0]), fb.astensor([1, 0, 0, 0])) == 0
        assert (
            fb.fnr(
                fb.astensor([1, 1, 0, 0]),
                fb.astensor([1, 1, 1, 0]),
                fb.astensor([1, 1, 0, 1]),
            )
            == 0
        )


def test_frr():
    for _ in environment():
        assert fb.frr(fb.astensor([1, 1, 0, 0]), fb.astensor([1, 1, 1, 0])) > 0
        assert fb.frr(fb.astensor([1, 1, 0, 0]), fb.astensor([1, 0, 0, 0])) == 0
        assert (
            fb.fnr(
                fb.astensor([1, 1, 0, 0]),
                fb.astensor([1, 1, 1, 0]),
                fb.astensor([1, 1, 0, 1]),
            )
            == 0
        )


def test_far():
    for _ in environment():
        assert fb.far(fb.astensor([1, 1, 0, 0]), fb.astensor([1, 1, 1, 0])) == 0
        assert fb.far(fb.astensor([1, 1, 1, 0]), fb.astensor([1, 1, 0, 0])) > 0
        assert (
            fb.far(
                fb.astensor([1, 1, 1, 0]),
                fb.astensor([1, 1, 0, 0]),
                fb.astensor([1, 1, 0, 1]),
            )
            == 0
        )


def test_auc():
    for _ in environment():
        assert (
            fb.auc(
                scores=fb.astensor([0.5, 0.8, 0.3, 0.2]),
                labels=fb.astensor([1, 1, 1, 0]),
            )
            > 0.5
        )


def test_f1k():
    for _ in environment():
        assert (
            fb.metrics.topf1(
                scores=fb.astensor([0.5, 0.8, 0.3, 0.2, 0, 0.1, 0.12]),
                labels=fb.astensor([1, 1, 1, 0, 0, 0, 0]),
            )
            == 1
        )


def test_hr():
    for _ in environment():
        assert (
            fb.metrics.tophr(
                scores=fb.astensor([0.5, 0.8, 0.3, 0.2, 0, 0.1, 0.12]),
                labels=fb.astensor([1, 1, 1, 0, 0, 0, 0]),
            )
            == 1
        )


def test_reck():
    for _ in environment():
        assert (
            fb.metrics.toprec(
                scores=fb.astensor([0.5, 0.8, 0.3, 0.2, 0, 0.1, 0.12]),
                labels=fb.astensor([1, 1, 1, 0, 0, 0, 0]),
            )
            == 1
        )


def test_pinball():
    for _ in environment():
        mae = fb.metrics.mae(
            scores=fb.astensor([0.5, 0.8, 0.3, 0.2]), targets=fb.astensor([1, 1, 1, 0])
        )
        pinball = fb.metrics.pinball(
            scores=fb.astensor([0.5, 0.8, 0.3, 0.2]),
            targets=fb.astensor([1, 1, 1, 0]),
            alpha=0.5,
        )
        assert pinball == 0.5 * mae


def test_r2():
    for _ in environment():
        assert (
            fb.r2(
                scores=fb.astensor([0.5, 0.8, 0.3, 0.2]),
                targets=fb.astensor([0.45, 0.85, 0.4, 0.1]),
            )
            > 0
        )


def test_mae():
    for _ in environment():
        assert (
            fb.mae(scores=fb.astensor([0.5, 0.8]), targets=fb.astensor([0, 1])) == 0.35
        )


def test_mse():
    for _ in environment():
        assert (
            fb.mse(scores=fb.astensor([0.5, 0.8]), targets=fb.astensor([0, 1])) == 0.145
        )


def test_rmse():
    for _ in environment():
        assert (
            abs(
                fb.rmse(scores=fb.astensor([0.5, 0.8]), targets=fb.astensor([0, 1]))
                - 0.3807886552931954
            )
            < 1.0e-6
        )
