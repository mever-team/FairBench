from fairbench.fork import parallel
import scipy, numpy as np


@parallel
def multiplication(predictions, sensitive, max_sensitive=1):
    non_sensitive = max_sensitive - sensitive
    sum_sensitive = sensitive.sum()
    sum_non_sensitive = non_sensitive.sum()
    assert sum_sensitive != 0 and sum_non_sensitive != 0
    r1 = (predictions * sensitive).sum() / sum_sensitive
    r2 = (predictions * non_sensitive).sum() / sum_non_sensitive
    assert r1 != 0 and r2 != 0
    return predictions * sensitive / r1 + predictions * non_sensitive / r2


@parallel
def skew(predictions, y, sensitive, culep_params=(0, 0, 0, 0), max_sensitive=1):
    p = culep_params
    error = (y - predictions).abs()
    sample_weight_sensitive = (
        p[0] * (error * p[1]).exp() + (1 - p[0]) * (-error * p[1]).exp()
    )
    sample_weight_non_sensitive = (
        p[2] * (error * p[3]).exp() + (1 - p[2]) * (-error * p[3]).exp()
    )
    return (
        sensitive * sample_weight_sensitive
        + (max_sensitive - sensitive) * sample_weight_non_sensitive
    )


@parallel
def culep(yscores, y, sensitive, objective, skew=skew):
    last_predictions = yscores()
    best_p = None
    for _ in range(1):
        sample_weight = skew(last_predictions, y, sensitive)
        best_p = scipy.optimize.minimize(
            lambda p: -objective(sample_weight=sample_weight(culep_params=p)),
            np.array([0.5, 0.5, 0, 0]),
            method="Nelder-Mead",
            bounds=((0, 1), (0, 1), (-3, 3), (-3, 3)),
        ).x
        last_predictions = yscores(sample_weight=sample_weight(culep_params=best_p))
    sample_weight = skew(last_predictions, y, sensitive)
    return yscores.aspects(sample_weight=sample_weight(culep_params=best_p))
