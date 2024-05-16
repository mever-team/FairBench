import fairbench as fb
import numpy as np

test, y, yhat = fb.demos.adult()
indexes = np.random.choice(np.arange(y.size), 200)


sensitive = fb.Fork(fb.categories @ test[9], fb.categories @ test[8]).intersectional()
report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
print(sensitive)
print(report.wmean.pr)


test = test.iloc[indexes]
y = y[indexes]
yhat = yhat[indexes]
sensitive = (
    fb.Fork(fb.categories @ test[9], fb.categories @ test[8]).relax().intersectional()
)
report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
print(sensitive)
print(report.wmean.pr)
