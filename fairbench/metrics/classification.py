from fairbench.fork import parallel
from eagerpy import Tensor


@parallel
def accuracy(predictions: Tensor, labels: Tensor):
    return 1 - (predictions - labels).abs().mean()


@parallel
def pr(predictions: Tensor, sensitive: Tensor):
    sum_sensitive = sensitive.sum()
    if sum_sensitive == 0:
        return sum_sensitive
    return (predictions * sensitive).sum() / sum_sensitive


@parallel
def positives(predictions: Tensor, sensitive: Tensor):
    return (predictions * sensitive).mean()


@parallel
def fpr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Tensor,
):
    error = (predictions - labels).abs() * predictions
    error_sensitive = error * sensitive
    num_sensitive = (sensitive * predictions).sum()
    return error_sensitive.sum() / num_sensitive


@parallel
def fnr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Tensor,
    max_prediction: float = 1,
):
    negatives = max_prediction - predictions
    error = (predictions - labels).abs() * negatives
    error_sensitive = error * sensitive
    num_sensitive = (sensitive * negatives).sum()
    if num_sensitive == 0:
        return num_sensitive
    return error_sensitive.sum() / num_sensitive
