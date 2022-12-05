from fairbench.utils import SHARED, GENERATOR, framework, multimodal
import numpy
import scipy


@framework
def multiplication(predictions,
                   sensitive=SHARED,
                   max_sensitive=1,
                   backend=numpy):
    non_sensitive = max_sensitive - sensitive
    sum_sensitive = backend.sum(sensitive)
    sum_non_sensitive = backend.sum(non_sensitive)
    assert sum_sensitive != 0 and sum_non_sensitive != 0
    r1 = backend.sum(predictions*sensitive) / sum_sensitive
    r2 = backend.sum(predictions*non_sensitive) / sum_non_sensitive
    assert r1 != 0 and r2 != 0
    return predictions*sensitive/r1 + predictions*non_sensitive/r2


@framework
def skew(predictions,
         y,
         sensitive,
         culep_params=(0, 0, 0, 0),
         max_sensitive=1,
         backend=numpy):
    p = culep_params
    error = backend.abs(y - predictions)
    sample_weight_sensitive = p[0] * backend.exp(error * p[1]) + (1 - p[0]) * backend.exp(-error * p[1])
    sample_weight_non_sensitive = p[2] * backend.exp(error * p[3]) + (1 - p[2]) * backend.exp(-error * p[3])
    return sensitive * sample_weight_sensitive + (max_sensitive - sensitive) * sample_weight_non_sensitive


@multimodal
def culep(yscores, y, sensitive, objective, skew=skew):
    last_predictions = yscores()
    best_p = None
    for _ in range(1):
        sample_weight = skew(last_predictions, y, sensitive)
        best_p = scipy.optimize.minimize(lambda p: -objective(sample_weight=sample_weight(culep_params=p)),
                                         numpy.array([0.5, 0.5, 0, 0]),
                                         method='Nelder-Mead',
                                         bounds=((0,1), (0, 1), (-3, 3), (-3, 3))).x
        last_predictions = yscores(sample_weight=sample_weight(culep_params=best_p))
    sample_weight = skew(last_predictions, y, sensitive)
    return yscores.aspects(sample_weight=sample_weight(culep_params=best_p))

