import fairbench as fb
from .test_forks import environment


def test_modelcards(monkeypatch):
    import webbrowser

    for _ in environment():
        monkeypatch.setattr(webbrowser, "open_new", lambda: None)
        for setting, protected in [(fb.demos.adult, 8), (fb.demos.bank, "marital")]:
            test, y, yhat = setting()
            sensitive = fb.Fork(fb.categories @ test[protected])
            report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
            assert "prule" in fb.stamps.available()
            stamps = fb.combine(
                fb.stamps.prule(report),
                fb.stamps.accuracy(report),
                fb.stamps.dfpr(report),
                fb.stamps.dfnr(report),
                fb.stamps.four_fifths(report),
            )

            fb.modelcards.toyaml(stamps)  # TODO: add this to texts
            texts = [fb.modelcards.tohtml(stamps), fb.modelcards.tomarkdown(stamps)]

            for text in texts:
                assert "Metrics" in text
                assert "Caveats and Recommendations" in text
                assert "Factors" in text
