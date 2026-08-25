from fairbench import v1 as fb
from .test_forks import environment


def test_settings(monkeypatch):
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    for _ in environment():
        for setting, protected in [
            (fb.bench.tabular.adult, 8),
            (fb.bench.tabular.bank, "marital"),
            (fb.bench.tabular.compas, "sex"),
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

    monkeypatch.setattr(plt, "show", lambda: None)

    for _ in environment():
        for setting, protected in [
            (fb.bench.tabular.adult, 9),
            (fb.bench.tabular.bank, "marital"),
            (fb.bench.tabular.compas, "sex"),
        ]:
            # monkeypatch.setattr(plt, "show", lambda: None)
            test, y, yhat = setting(predict="probabilities")
            s = fb.Fork(fb.categories @ test[protected])

            report = fb.multireport(scores=yhat, labels=y, sensitive=s)
            fb.visualize(report.min.auc.explain.explain)
            fb.text_visualize(report.min.auc.explain.explain)


def test_data_loading():
    for setting in (
        fb.bench.tabular.adult,
        fb.bench.tabular.bank,
        fb.bench.tabular.compas,
    ):
        assert len(setting(predict="data")) == 6  # x_train, y_train, x, y, train, test
