import fairbench as fb
from .test_forks import environment


def test_settings(monkeypatch):
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    for _ in environment():
        for setting, protected in [
            (fb.demos.adult, 8),
            (fb.demos.bank, "marital"),
            (fb.demos.compas, "sex"),
        ]:
            test, y, yhat = setting()
            sensitive = fb.Fork(fb.categories @ test[protected])
            report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
            fb.visualize(report)
            fb.visualize(report.min.accuracy.explain.explain)
            fb.visualize(report.min.accuracy.explain)
            fb.visualize(report.min.explain)


def test_curve_visualization(monkeypatch):
    from matplotlib import pyplot as plt

    for _ in environment():
        for setting, protected in [
            (fb.demos.adult, 9),
            (fb.demos.bank, "marital"),
            (fb.demos.compas, "sex"),
        ]:
            monkeypatch.setattr(plt, "show", lambda: None)
            test, y, yhat = setting(predict="probabilities")
            s = fb.Fork(fb.categories @ test[protected])

            report = fb.multireport(scores=yhat, labels=y, sensitive=s)
            fb.visualize(report.min.auc.explain.explain)
