from fairbench import v1 as fb
import numpy as np
from .test_forks import environment


def test_batch_accumulation():
    for _ in environment():
        import torch
        from sklearn import linear_model

        fb.setbackend("torch")
        x = [
            [0, 0.1],
            [0.9, 0],
            [0, 0.1],
            [1.1, -0.1],
            [0.1, 0.1],
            [0.9, 0.1],
            [0.4, 0.5],
            [0.6, 0.3],
        ]
        y = [0, 1, 0, 1, 0, 1, 1, 1]
        s = [0, 0, 0, 0, 1, 1, 1, 1]

        x, y, s = (
            torch.from_numpy(np.array(x)),
            torch.from_numpy(np.array(y)),
            torch.from_numpy(np.array(s)),
        )
        s2 = [0, 1, 1, 1, 1, 1, 1, 1]
        s3 = np.array(s2)
        # s = 1-s2  # really hard imbalance in the sensitive (isecreport handles this)
        s2 = 1 - s
        sensitive = fb.Fork(gender=s, ehtnicity=s2, ethinicity2=s3)

        classifier = linear_model.LogisticRegression(max_iter=1000)
        classifier = classifier.fit(x, y)
        yhat = classifier.predict(x)

        vals = None
        vals = fb.concatenate(
            vals, fb.todict(predictions=yhat, labels=y, sensitive=sensitive)
        )
        vals = fb.concatenate(
            vals, fb.todict(predictions=yhat, labels=y, sensitive=sensitive)
        )

        report = fb.isecreport(vals)
        fb.describe(report)
        assert report.minprule.bayesian.value > 0.4
        fb.setbackend("numpy")
        report = fb.multireport(vals)
        fb.describe(report)
        assert report.minratio.pr.value > 0.4
