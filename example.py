import fairbench as fb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from timeit import default_timer as time


def load():
    x = [[0, 0.1], [0.9, 0], [0, 0.1], [1.1, -0.1], [0.1, 0.1], [0.9, 0.1], [0.4, 0.5], [0.6, 0.3]]
    y = [0, 1, 0, 1, 0, 1, 1, 1]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    return x, y, s


if __name__ == '__main__':  # this is necessary to instantiate the distributed environment
    tic = time()
    #fb.distributed()

    x, y, s = load()
    x, y, s = np.array(x), np.array(y), np.array(s)
    s2 = [0, 1, 1, 0, 1, 1, 1, 1]
    s2 = np.array(s2)

    sensitive = fb.Fork(case1=s, case2=s2)
    classifier = fb.Fork(case1=LogisticRegression(), case2=MLPClassifier())
    classifier = classifier.fit(x, y)
    yhat = classifier.predict(x)
    yhat = (yhat.case1+yhat.case2)/2

    vals = None
    vals = fb.concatenate(vals, fb.kwargs(predictions=yhat, labels=y, sensitive=sensitive))

    """
    report = fb.multireport(vals)
    print(fb.tojson(report))
    fb.describe(report)
    print('ETA', time()-tic)
    fb.visualize(report)
    """

    report = fb.accreport(vals)

    mean_across_branches = fb.reduce(report, fb.mean, name="avg")
    max_abs_across_branches = fb.reduce(report, fb.budget, expand=fb.ratio, transform=fb.abs)
    new_report = fb.combine(report, mean_across_branches, max_abs_across_branches)

    fb.describe(new_report)

