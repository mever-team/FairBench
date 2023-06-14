from fairbench.bench.loader import read_csv, features
import sklearn


def adult(classifier=sklearn.linear_model.LogisticRegression(max_iter=1000).fit,
          scaler=sklearn.preprocessing.MinMaxScaler().fit_transform,
          predict="predict"):
    train = read_csv("adult/adult.data", header=None, skipinitialspace=True)
    test = read_csv("adult/adult.test", header=None, skipinitialspace=True, skiprows=[0])
    numeric = [0, 4, 11, 12]
    categorical = [1, 3, 5, 6]
    x_train = features(train, numeric, categorical)
    y_train = (train[14] == ">50K").values
    x = features(test, numeric, categorical)
    y = (test[14] == ">50K.").values
    x_train = scaler(x_train)
    x = scaler(x)
    classifier = classifier(x_train, y_train)
    if predict == "predict":
        yhat = classifier.predict(x)
    elif predict == "probabilities":
        yhat = classifier.predict_proba(x)[:,0]
    else:
        raise NotImplementedError()
    return test, y, yhat
