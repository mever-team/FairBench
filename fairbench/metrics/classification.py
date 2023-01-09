from fairbench.modal import multimodal
from eagerpy import Tensor


@multimodal
def accuracy(predictions: Tensor, labels: Tensor):
    return 1 - (predictions - labels).abs().mean()
