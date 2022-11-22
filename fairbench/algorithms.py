from fairbench.utils import SHARED, GENERATOR, framework
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
def culep(predictions=GENERATOR,
          objective=GENERATOR,
          sensitive=SHARED,
          ground_truth=SHARED,
          max_sensitive=1,
          backend=numpy):
    last_predictions = predictions()
    for _ in range(1):
        error = backend.abs(ground_truth - last_predictions)

        def skew(p):
            sample_weight_sensitive = p[0]*backend.exp(error*p[1]) + (1-p[0])*backend.exp(-error*p[1])
            sample_weight_non_sensitive = p[2]*backend.exp(error*p[3]) + (1-p[2])*backend.exp(-error*p[3])
            return sensitive*sample_weight_sensitive + (max_sensitive-sensitive)*sample_weight_non_sensitive

        best_p = scipy.optimize.minimize(lambda p: -objective(sample_weight=skew(p)),
                                         numpy.array([0.5, 0.5, 0, 0]),
                                         method='Nelder-Mead',
                                         bounds=((0,1), (0, 1), (-3, 3), (-3, 3)))
        last_predictions = predictions(sample_weight=skew(best_p.x))
    return last_predictions

