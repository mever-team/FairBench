import fairbench
import fairbench as fb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from timeit import default_timer as time
import torch

fairbench.setbackend("torch")


def load():
    x = [[0, 0.1], [0.9, 0], [0, 0.1], [1.1, -0.1], [0.1, 0.1], [0.9, 0.1], [0.4, 0.5], [0.6, 0.3]]
    y = [0, 1, 0, 1, 0, 1, 1, 1]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    return x, y, s


if __name__ == '__main__':  # this is necessary to instantiate the distributed environment
    tic = time()
    #fb.distributed()

    x, y, s = load()
    x, y, s = torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y)), torch.from_numpy(np.array(s))
    s2 = [0, 1, 1, 1, 1, 1, 1, 1]
    s2 = np.array(s2)

    sensitive = fb.Fork(case1=s, case2=s2)
    classifier = LogisticRegression()
    classifier = classifier.fit(x, y)
    yhat = classifier.predict(x)

    vals = None
    vals = fb.concatenate(vals, fb.kwargs(predictions=yhat, labels=y, sensitive=sensitive))

    print('ETA', time()-tic)

    #report = fb.isecreport(vals)
    #fb.visualize(report)
    #fb.visualize(report.bayesian["minprule"].explain)

    report = fb.multireport(vals)
    fb.describe(report)
    fb.visualize(report)
    fb.visualize(report.mean["accuracy"].explain)

