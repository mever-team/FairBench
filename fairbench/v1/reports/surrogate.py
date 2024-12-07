from fairbench.v1.blocks import framework, todata, identical
from fairbench.v1.core import Fork, multibranch_tensors
import numpy as np


# TODO: LR with 10 iterations to stop overtraining is a half-measure, switch to something stable
@multibranch_tensors
def surrogate_positives(predictions, sensitive, surrogate_model=None):
    if surrogate_model is None:
        from fairbench.fallbacks import LogisticRegression

        surrogate_model = LogisticRegression(max_iter=10)
    predictions = np.round(framework.reduce(predictions, identical, name=None).numpy())
    X = framework.reduce(sensitive, todata, name=None).numpy()
    surrogate_model = surrogate_model.fit(X, predictions)
    prediction_branches = dict()
    for branches in sensitive.iterate_intersections():
        Xbranch = np.array(
            [[1 if branch in branches else 0 for branch in sensitive._branches]]
        )
        yhat = float(surrogate_model.predict_proba(Xbranch)[:, 1][0])
        prediction_branches["&".join(branches)] = yhat
    return Fork(prediction_branches)
