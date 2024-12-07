from fairbench.v1.core import parallel, unit_bounded
from eagerpy import Tensor
from typing import Optional


@parallel
@unit_bounded
def dfpr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Tensor,
    non_sensitive: Optional[Tensor] = None,
):
    predictions, labels = (
        labels,
        predictions,
    )  # temporary fix: original implementation had them flipped
    if non_sensitive is None:
        non_sensitive = 1.0 - sensitive
    error = (predictions - labels).abs() * predictions
    error_sensitive = error * sensitive
    error_non_sensitive = error * non_sensitive
    num_sensitive = (sensitive * predictions).sum()
    num_non_sensitive = (non_sensitive * predictions).sum()
    if num_sensitive == 0 or num_non_sensitive == 0:
        return num_sensitive * 0
    return (
        error_sensitive.sum() / num_sensitive
        - error_non_sensitive.sum() / num_non_sensitive
    )


@parallel
@unit_bounded
def dfnr(
    predictions: Tensor,
    labels: Tensor,
    sensitive: Tensor,
    non_sensitive: Optional[Tensor] = None,
    # max_prediction: float = 1,
):
    predictions, labels = (
        labels,
        predictions,
    )  # temporary fix: original implementation had them flipped
    negatives = 1.0 - predictions
    if non_sensitive is None:
        non_sensitive = 1.0 - sensitive
    error = (predictions - labels).abs() * negatives
    error_sensitive = error * sensitive
    error_non_sensitive = error * non_sensitive
    num_sensitive = (sensitive * negatives).sum()
    num_non_sensitive = (non_sensitive * negatives).sum()
    if num_sensitive == 0 or num_non_sensitive == 0:
        return num_sensitive * 0
    return (
        error_sensitive.sum() / num_sensitive
        - error_non_sensitive.sum() / num_non_sensitive
    )
