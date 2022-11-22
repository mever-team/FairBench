from fairbench.utils import SHARED, framework
import numpy


@framework
def prule(predictions,
          sensitive=SHARED,
          max_sensitive=1,
          backend=numpy):
    non_sensitive = max_sensitive-sensitive
    sum_sensitive = backend.sum(sensitive)
    sum_non_sensitive = backend.sum(non_sensitive)
    if sum_sensitive == 0 or sum_non_sensitive == 0:
        return 0
    r1 = backend.sum(predictions*sensitive) / sum_sensitive
    r2 = backend.sum(predictions*non_sensitive) / sum_non_sensitive
    max_r = max(r1, r2)
    if max_r == 0:
        return max_r
    return min(r1, r2) / max_r


@framework
def accuracy(predictions,
             ground_truth=SHARED,
             backend=numpy):
    return 1-backend.mean(backend.abs(predictions-ground_truth))


@framework
def dfpr(predictions,
         sensitive=SHARED,
         ground_truth=SHARED,
         max_sensitive=1,
         backend=numpy):
    non_sensitive = max_sensitive-sensitive
    error = backend.abs(predictions-ground_truth)*predictions
    error_sensitive = error*sensitive
    error_non_sensitive = error*non_sensitive
    num_sensitive = backend.sum(sensitive*predictions)
    num_non_sensitive = backend.sum(non_sensitive*predictions)
    return backend.sum(error_sensitive)/num_sensitive - backend.sum(error_non_sensitive)/num_non_sensitive


@framework
def dfnr(predictions,
         sensitive=SHARED,
         ground_truth=SHARED,
         max_prediction=1,
         max_sensitive=1,
         backend=numpy):
    negatives = max_prediction-predictions
    non_sensitive = max_sensitive-sensitive
    error = backend.abs(predictions-ground_truth)*negatives
    error_sensitive = error*sensitive
    error_non_sensitive = error*non_sensitive
    num_sensitive = backend.sum(sensitive*negatives)
    num_non_sensitive = backend.sum(non_sensitive*negatives)
    if num_sensitive == 0 or num_non_sensitive == 0:
        return 0
    return backend.sum(error_sensitive)/num_sensitive - backend.sum(error_non_sensitive)/num_non_sensitive
