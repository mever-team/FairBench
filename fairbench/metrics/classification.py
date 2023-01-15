from fairbench.forks import parallel, Explainable
from eagerpy import Tensor


@parallel
def accuracy(predictions: Tensor, labels: Tensor, sensitive: Tensor = None):
    if sensitive is None:
        sensitive = predictions.ones_like()
    num_sensitive = sensitive.sum()
    if num_sensitive == 0:
        return 0
    return 1 - ((predictions - labels) * sensitive).abs().sum() / num_sensitive


@parallel
def pr(predictions: Tensor, sensitive: Tensor = None):
    if sensitive is None:
        sensitive = predictions.ones_like()
    sum_sensitive = sensitive.sum()
    if sum_sensitive == 0:
        return sum_sensitive
    return (predictions * sensitive).sum() / sum_sensitive


@parallel
def positives(predictions: Tensor, sensitive: Tensor):
    return (predictions * sensitive).mean()


@parallel
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
    if num_sensitive == 0:
        return 0
    return error_sensitive.sum() / num_sensitive


@parallel
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
    if num_sensitive == 0:
        return 0
    return error_sensitive.sum() / num_sensitive
