import fairbench as fb
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression


train = fb.read_csv("adult/adult.data", header=None, skipinitialspace=True)
test = fb.read_csv("adult/adult.test", header=None, skipinitialspace=True, skiprows=[0])
numeric = [0, 4, 11, 12]
categorical = [1, 3, 5, 6]
x_train = fb.features(train, numeric, categorical)
y_train = (train[14]==">50K").values
x_test = fb.features(test, numeric, categorical)
y_test = (test[14]==">50K.").values


x_train_scaled = sklearn.preprocessing.MinMaxScaler().fit_transform(x_train)
x_test_scaled = sklearn.preprocessing.MinMaxScaler().fit_transform(x_test)

classifier = LogisticRegression(max_iter=1000)
classifier = classifier.fit(x_train_scaled, y_train)
predictions = classifier.predict(x_test_scaled)
sensitive = fb.Fork(gender=fb.categories@test[9])

fair_predictions = fb.mitigation.multiplication(predictions, sensitive)
reduced_predictions = fb.areduce(fair_predictions, fb.mean)
reduced_predictions = reduced_predictions/reduced_predictions.max()

fb.describe(fb.multireport(predictions=predictions, labels=y_test, sensitive=sensitive))
fb.describe(fb.multireport(predictions=reduced_predictions, labels=y_test, sensitive=sensitive))
