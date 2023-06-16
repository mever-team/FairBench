from fairbench.forks import parallel, unit_bounded
from fairbench.forks.explanation import Explainable, ExplanationCurve
from eagerpy import Tensor


@parallel
@unit_bounded
def accuracy(predictions: Tensor, labels: Tensor, sensitive: Tensor = None):
    if sensitive is None:
        sensitive = predictions.ones_like()
    num_sensitive = sensitive.sum()
    true = ((predictions - labels) * sensitive).abs().sum()
    return Explainable(
        0 if num_sensitive == 0 else 1 - true / num_sensitive,
        samples=num_sensitive,
        true=true,
    )


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


@parallel
@unit_bounded
def pr(predictions: Tensor, sensitive: Tensor = None):
    if sensitive is None:
        sensitive = predictions.ones_like()
    sum_sensitive = sensitive.sum()
    sum_positives = (predictions * sensitive).sum()
    return Explainable(
        0 if sum_sensitive == 0 else (sum_positives / sum_sensitive),
        samples=sum_sensitive,
        positives=sum_positives,
    )


@parallel
@unit_bounded
def positives(predictions: Tensor, sensitive: Tensor):
    return (predictions * sensitive).sum()


@parallel
@unit_bounded
def tpr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Tensor = None,
    max_prediction: float = 1,
):
    if sensitive is None:
        sensitive = predictions.ones_like()
    error = (max_prediction - (predictions - labels).abs()) * predictions
    error_sensitive = error * sensitive
    num_sensitive = (sensitive * predictions).sum()
    return Explainable(
        0 if num_sensitive == 0 else (error_sensitive.sum() / num_sensitive),
        positives=num_sensitive,
        true_positives=error_sensitive.sum(),
        samples=sensitive.sum(),
    )


@parallel
@unit_bounded
def fpr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Tensor = None,
):
    if sensitive is None:
        sensitive = predictions.ones_like()
    error = (predictions - labels).abs() * predictions
    error_sensitive = error * sensitive
    num_sensitive = (sensitive * predictions).sum()
    if num_sensitive == 0:
        return 0
    return Explainable(
        0 if num_sensitive == 0 else (error_sensitive.sum() / num_sensitive),
        positives=num_sensitive,
        false_positives=error_sensitive.sum(),
        samples=sensitive.sum(),
    )


@parallel
@unit_bounded
def tnr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Tensor = None,
    max_prediction: float = 1,
):
    if sensitive is None:
        sensitive = predictions.ones_like()
    negatives = max_prediction - predictions
    error = (max_prediction - (predictions - labels).abs()) * negatives
    error_sensitive = error * sensitive
    num_sensitive = (sensitive * negatives).sum()
    return Explainable(
        0 if num_sensitive == 0 else (error_sensitive.sum() / num_sensitive),
        negatives=num_sensitive,
        true_negatives=error_sensitive.sum(),
        samples=sensitive.sum(),
    )


@parallel
@unit_bounded
def fnr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Tensor = None,
    max_prediction: float = 1,
):
    if sensitive is None:
        sensitive = predictions.ones_like()
    negatives = max_prediction - predictions
    error = (predictions - labels).abs() * negatives
    error_sensitive = error * sensitive
    num_sensitive = (sensitive * negatives).sum()
    return Explainable(
        0 if num_sensitive == 0 else (error_sensitive.sum() / num_sensitive),
        negatives=num_sensitive,
        false_negatives=error_sensitive.sum(),
        samples=sensitive.sum(),
    )
