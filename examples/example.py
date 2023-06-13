import fairbench as fb
import sklearn
from sklearn.linear_model import LogisticRegression


train = fb.read_csv("adult/adult.data", header=None, skipinitialspace=True)
test = fb.read_csv("adult/adult.test", header=None, skipinitialspace=True, skiprows=[0])
numeric = [0, 4, 11, 12]
categorical = [1, 3, 5, 6]
x_train = fb.features(train, numeric, categorical)
y_train = (train[14]==">50K").values
x = fb.features(test, numeric, categorical)
y = (test[14]==">50K.").values


x_train = sklearn.preprocessing.MinMaxScaler().fit_transform(x_train)
x = sklearn.preprocessing.MinMaxScaler().fit_transform(x)

classifier = LogisticRegression(max_iter=1000)
classifier = classifier.fit(x_train, y_train)
yhat = classifier.predict(x)
s = fb.Fork(fb.categories@test[9], fb.categories@test[8])
#print(s.sum())

report = fb.multireport(predictions=yhat, labels=y, sensitive=s)
print(report.min.tpr.explain.explain.true_positives)
