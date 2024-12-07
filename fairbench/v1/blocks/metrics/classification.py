from fairbench.v1.core import parallel, unit_bounded, role
from fairbench.v1.core import Explainable
from eagerpy import Tensor
from typing import Optional


@role("metric")
@parallel
def accuracy(
    predictions: Tensor, labels: Tensor, sensitive: Optional[Tensor] = None
) -> Explainable:
    if sensitive is None:
        sensitive = predictions.ones_like()
    num_sensitive = sensitive.sum()
    true = ((predictions == labels) * sensitive).abs().sum()
    return Explainable(
        0 if num_sensitive == 0 else true / num_sensitive,
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
    desired_positives = (labels * sensitive).sum()
    tpr_value = 0 if desired_positives == 0 else true_positives / desired_positives
    return Explainable(
        tpr_value,
        desired_positives=desired_positives.item(),
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
    desired_positives = (labels * sensitive).sum()
    fpr_value = 1 if desired_positives == 0 else false_positives / desired_positives
    return Explainable(
        fpr_value,
        desired_positives=desired_positives.item(),
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
    desired_negatives = ((max_prediction - labels) * sensitive).sum()
    tnr_value = 0 if desired_negatives == 0 else true_negatives / desired_negatives
    return Explainable(
        tnr_value,
        desired_negatives=desired_negatives.item(),
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
    desired_negatives = ((max_prediction - labels) * sensitive).sum()
    fnr_value = 1 if desired_negatives == 0 else false_negatives / desired_negatives
    return Explainable(
        fnr_value,
        desired_negatives=desired_negatives.item(),
        false_negatives=false_negatives.item(),
        samples=sensitive.sum().item(),
    )


@role("metric")
@parallel
@unit_bounded
def far(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Optional[Tensor] = None,
    max_prediction: float = 1,
):
    if sensitive is None:
        sensitive = predictions.ones_like()
    false_positives = (predictions * (max_prediction - labels) * sensitive).sum()
    samples = sensitive.sum()
    far_value = 0 if samples == 0 else false_positives / samples
    return Explainable(
        far_value,
        false_positives=false_positives.item(),
        samples=samples.item(),
    )


@role("metric")
@parallel
@unit_bounded
def frr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Optional[Tensor] = None,
    max_prediction: float = 1,
):
    if sensitive is None:
        sensitive = predictions.ones_like()
    false_negatives = ((max_prediction - predictions) * labels * sensitive).sum()
    samples = sensitive.sum()
    frr_value = 1 if positives == 0 else false_negatives / samples
    return Explainable(
        frr_value,
        false_negatives=false_negatives.item(),
        samples=samples.item(),
    )
