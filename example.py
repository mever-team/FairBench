import fairbench as fb
from sklearn.linear_model import LogisticRegression


def load():
    x = [[0, 0.1], [0.9, 0], [0, 0.1], [1.1, -0.1], [0.1, 0.1], [0.9, 0.1], [0.4, 0.5], [0.6, 0.3]]
    y = [0, 1, 0, 1, 0, 1, 1, 1]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    return x, y, s


x, y, sensitive = load()

x = fb.array(x)
y = fb.array(y)
sensitive = fb.array(sensitive)

classifier = fb.instance(LogisticRegression)
classifier = fb.fit(classifier)
yhat = classifier.predict(x)
yhat = fb.culep(yhat, fb.accuracy(yhat)+fb.prule(yhat))

#print(fb.prule(yhat)(sensitive=sensitive))
print(fb.report(yhat, features=x, ground_truth=y, sensitive=sensitive))
