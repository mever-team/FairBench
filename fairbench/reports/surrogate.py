from fairbench.blocks import framework, todata, identical
from fairbench.core.fork import Fork, multibranch_tensors
from sklearn.linear_model import LogisticRegression
import numpy as np


@multibranch_tensors
def surrogate_positives(
    predictions, sensitive, surrogate_model=LogisticRegression(max_iter=1000)
):
    predictions = np.round(framework.reduce(predictions, identical, name=None).numpy())
    X = framework.reduce(sensitive, todata, name=None).numpy()
    surrogate_model = surrogate_model.fit(X, predictions)
    prediction_branches = dict()
    for branches in sensitive.iterate_intersections():
        Xbranch = np.array(
            [[1 if branch in branches else 0 for branch in sensitive._branches]]
        )
        yhat = float(surrogate_model.predict_proba(Xbranch)[:, 1])
        prediction_branches["&".join(branches)] = yhat
    return Fork(prediction_branches)
