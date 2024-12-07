from fairbench.v1.core import parallel, unit_bounded, role
from fairbench.v1.core import Explainable, ExplanationCurve
from eagerpy import Tensor


@role("metric")
@parallel
def max_error(scores: Tensor, targets: Tensor, sensitive: Tensor = None):
    return Explainable(((scores - targets) * sensitive).abs().sum())


@role("metric")
@parallel
def mae(scores: Tensor, targets: Tensor, sensitive: Tensor = None):
    if sensitive is None:
        sensitive = scores.ones_like()
    num_sensitive = sensitive.sum()
    true = ((scores - targets) * sensitive).abs().sum()
    return Explainable(
        0 if num_sensitive == 0 else true / num_sensitive,
        samples=num_sensitive,
        sae=true,
    )


@role("metric")
@parallel
def rmse(scores: Tensor, targets: Tensor, sensitive: Tensor = None):
    if sensitive is None:
        sensitive = scores.ones_like()
    num_sensitive = sensitive.sum()
    true = ((scores - targets) ** 2 * sensitive).sum()
    return Explainable(
        0 if num_sensitive == 0 else (true / num_sensitive) ** 0.5,
        samples=num_sensitive,
        sse=true,
    )


@role("metric")
@parallel
def mse(scores: Tensor, targets: Tensor, sensitive: Tensor = None):
    if sensitive is None:
        sensitive = scores.ones_like()
    num_sensitive = sensitive.sum()
    true = ((scores - targets) ** 2 * sensitive).sum()
    return Explainable(
        0 if num_sensitive == 0 else (true / num_sensitive),
        samples=num_sensitive,
        sse=true,
    )


@role("metric")
@parallel
def r2(scores: Tensor, targets: Tensor, sensitive: Tensor = None, deg_freedom: int = 0):
    assert deg_freedom >= 0
    if sensitive is None:
        sensitive = scores.ones_like()
    num_sensitive = sensitive.sum()
    true = ((scores - targets) ** 2 * sensitive).sum()
    target_mean_squares = (targets**2 * sensitive).sum() / num_sensitive
    target_mean = (targets**2 * sensitive).sum() / num_sensitive
    target_variance = target_mean_squares - target_mean**2
    return Explainable(
        (
            0
            if num_sensitive == 0
            else (1 - (true / target_variance))
            * ((num_sensitive - 1) / (num_sensitive - 1 - deg_freedom))
        ),
        samples=num_sensitive,
        # target_mean=target_mean,
        # target_variance=target_variance,
        deg_freedom=deg_freedom,
        sse=true,
    )


@role("metric")
@parallel
def pinball(
    scores: Tensor, targets: Tensor, sensitive: Tensor = None, alpha: float = 0.5
):
    assert 0 <= alpha <= 1
    if sensitive is None:
        sensitive = scores.ones_like()
    num_sensitive = sensitive.sum()
    loss = alpha * (targets - scores).maximum(0) + (1 - alpha) * (
        scores - targets
    ).maximum(0)
    filtered = (loss * sensitive).sum()
    return Explainable(
        0 if num_sensitive == 0 else filtered / num_sensitive,
        samples=num_sensitive,
    )
