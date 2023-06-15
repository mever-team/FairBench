import fairbench as fb
from .test_forks import environment


def test_settings(monkeypatch):
    from matplotlib import pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)
    for _ in environment():
        for setting, protected in [(fb.demos.adult, 8), (fb.demos.bank, "marital")]:
            test, y, yhat = setting()
            sensitive = fb.Fork(fb.categories@test[protected])
            fb.visualize(fb.multireport(predictions=yhat, labels=y, sensitive=sensitive))
