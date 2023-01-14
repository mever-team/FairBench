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
    vals = fb.concatenate(vals, fb.kwargs(predictions=yhat, labels=y, sensitive=sensitive))

    fb.describe(fb.report(vals))
    print('ETA', time()-tic)
