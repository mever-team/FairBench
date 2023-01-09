from fairbench.fork import parallel
from eagerpy import Tensor


@parallel
def accuracy(predictions: Tensor, labels: Tensor):
    return 1 - (predictions - labels).abs().mean()
