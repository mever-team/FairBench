from fairbench.bench.loader import read_csv, features
import sklearn


def adult(
    classifier=sklearn.linear_model.LogisticRegression(max_iter=1000).fit,
    scaler=sklearn.preprocessing.MinMaxScaler().fit_transform,
    predict="predict",
):
    train = read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None,
        skipinitialspace=True,
    )
    test = read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        header=None,
        skipinitialspace=True,
        skiprows=[0],
    )
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
        yhat = classifier.predict_proba(x)[:, 1]
    else:
        raise NotImplementedError()
    return test, y, yhat


def bank(
    classifier=sklearn.linear_model.LogisticRegression(max_iter=1000).fit,
    scaler=sklearn.preprocessing.MinMaxScaler().fit_transform,
    predict="predict",
):
    data = read_csv(
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv",
        delimiter=";",
    )
    train, test = sklearn.model_selection.train_test_split(data)
    numeric = ["age", "duration", "campaign", "pdays", "previous"]
    categorical = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "poutcome",
    ]
    x_train = features(train, numeric, categorical)
    y_train = (train["y"] == "yes").values
    x = features(test, numeric, categorical)
    y = (test["y"] == "yes").values
    x_train = scaler(x_train)
    x = scaler(x)
    classifier = classifier(x_train, y_train)
    if predict == "predict":
        yhat = classifier.predict(x)
    elif predict == "probabilities":
        yhat = classifier.predict_proba(x)[:, 0]
    else:
        raise NotImplementedError()
    return test, y, yhat
