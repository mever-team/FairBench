from yamlres import Loader, Runner
import fairbench as fb
import sklearn


resource = "yamlres/reports.yaml"
reporter = Loader().load(resource)
reporter = Runner().init(reporter)

train = fb.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None, skipinitialspace=True)
test = fb.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", header=None, skipinitialspace=True, skiprows=[0])
numeric = [0, 4, 11, 12]
categorical = [1, 3, 5, 6]
x_train = fb.features(train, numeric, categorical)
y_train = (train[14]==">50K").values
x = fb.features(test, numeric, categorical)
y = (test[14]==">50K.").values
x_train = sklearn.preprocessing.MinMaxScaler().fit_transform(x_train)
x = sklearn.preprocessing.MinMaxScaler().fit_transform(x)
classifier = sklearn.linear_model.LogisticRegression(max_iter=1000)
classifier = classifier.fit(x_train, y_train)
yhat = classifier.predict(x)
s = fb.Fork(fb.categories@test[9], fb.categories@test[8])

report = reporter(predictions=yhat, labels=y, sensitive=s)

print(report)