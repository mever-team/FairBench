import fairbench as fb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def load():
    x = [[0, 0.1], [0.9, 0], [0, 0.1], [1.1, -0.1], [0.1, 0.1], [0.9, 0.1], [0.4, 0.5], [0.6, 0.3]]
    y = [0, 1, 0, 1, 0, 1, 1, 1]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    return x, y, s


x, y, s = load()
x, y, s = np.array(x), np.array(y), np.array(s)
s2 = [0, 1, 1, 0, 0, 1, 0, 1]
s2 = np.array(s2)

sensitive = fb.Modal(case1=s, case2=s2)
classifier = fb.Modal(case1=LogisticRegression(), case2=MLPClassifier())
classifier = classifier.fit(x, y)
yhat = classifier.predict(x)
yhat = (yhat.case1+yhat.case2)/2

fb.describe(fb.report(predictions=yhat, labels=y, sensitive=sensitive))


#fb.visualize(report)
