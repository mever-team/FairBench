from fairbench.core import parallel, unit_bounded, role
from fairbench.core.explanation import Explainable
from eagerpy import Tensor
from typing import Optional


@role("metric")
@parallel
@unit_bounded
def accuracy(
    predictions: Tensor, labels: Tensor, sensitive: Optional[Tensor] = None
) -> Explainable:
    if sensitive is None:
        sensitive = predictions.ones_like()
    num_sensitive = sensitive.sum()
    true = ((predictions - labels) * sensitive).abs().sum()
    return Explainable(
        0 if num_sensitive == 0 else 1 - true / num_sensitive,
        samples=num_sensitive,
        true=true,
    )


@role("metric")
@parallel
@unit_bounded
def pr(predictions: Tensor, sensitive: Optional[Tensor] = None):
    if sensitive is None:
        sensitive = predictions.ones_like()
    sum_sensitive = sensitive.sum()
    sum_positives = (predictions * sensitive).sum()
    pr_value = 0 if sum_sensitive == 0 else sum_positives / sum_sensitive
    return Explainable(
        pr_value,
        samples=sum_sensitive.item(),
        positives=sum_positives.item(),
    )


@role("metric")
@parallel
@unit_bounded
def positives(predictions: Tensor, sensitive: Optional[Tensor] = None):
    if sensitive is None:
        sensitive = predictions.ones_like()
    return Explainable((predictions * sensitive).sum(), samples=sensitive.sum())


@role("metric")
@parallel
@unit_bounded
def tpr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Optional[Tensor] = None,
):
    if sensitive is None:
        sensitive = predictions.ones_like()
    true_positives = (predictions * labels * sensitive).sum()
    positives = (labels * sensitive).sum()
    tpr_value = 0 if positives == 0 else true_positives / positives
    return Explainable(
        tpr_value,
        positives=positives.item(),
        true_positives=true_positives.item(),
        samples=sensitive.sum().item(),
    )


@role("metric")
@parallel
@unit_bounded
def fpr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Optional[Tensor] = None,
    max_prediction: float = 1,
):
    if sensitive is None:
        sensitive = predictions.ones_like()
    false_positives = (predictions * (max_prediction - labels) * sensitive).sum()
    negatives = ((max_prediction - labels) * sensitive).sum()
    fpr_value = 1 if negatives == 0 else false_positives / negatives
    return Explainable(
        fpr_value,
        negatives=negatives.item(),
        false_positives=false_positives.item(),
        samples=sensitive.sum().item(),
    )


@role("metric")
@parallel
@unit_bounded
def tnr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Optional[Tensor] = None,
    max_prediction: float = 1,
):
    if sensitive is None:
        sensitive = predictions.ones_like()
    true_negatives = (
        (max_prediction - predictions) * (max_prediction - labels) * sensitive
    ).sum()
    negatives = ((max_prediction - labels) * sensitive).sum()
    tnr_value = 0 if negatives == 0 else true_negatives / negatives
    return Explainable(
        tnr_value,
        negatives=negatives.item(),
        true_negatives=true_negatives.item(),
        samples=sensitive.sum().item(),
    )


@role("metric")
@parallel
@unit_bounded
def fnr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Optional[Tensor] = None,
    max_prediction: float = 1,
):
    if sensitive is None:
        sensitive = predictions.ones_like()
    false_negatives = ((max_prediction - predictions) * labels * sensitive).sum()
    positives = (labels * sensitive).sum()
    fnr_value = 1 if positives == 0 else false_negatives / positives
    return Explainable(
        fnr_value,
        positives=positives.item(),
        false_negatives=false_negatives.item(),
        samples=sensitive.sum().item(),
    )
