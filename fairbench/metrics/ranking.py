from fairbench.forks import parallel, unit_bounded, role
from fairbench.forks.explanation import Explainable, ExplanationCurve
from eagerpy import Tensor


@role("metric")
@parallel
@unit_bounded
def auc(scores: Tensor, labels: Tensor, sensitive: Tensor = None):
    import sklearn

    if sensitive is None:
        sensitive = scores.ones_like()
    scores = scores[sensitive == 1]
    labels = labels[sensitive == 1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels.numpy(), scores.numpy())
    return Explainable(
        sklearn.metrics.auc(fpr, tpr),
        curve=ExplanationCurve(fpr, tpr, "ROC"),
        samples=sensitive.sum(),
    )
