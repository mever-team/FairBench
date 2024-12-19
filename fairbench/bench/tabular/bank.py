from fairbench.bench.loader import read_csv, features
from fairbench.fallbacks import LogisticRegression, MinMaxScaler, train_test_split
from fairbench.bench.loader import cache


def bank(classifier=None, scaler=None, predict="predict", seed=None, test_size=0.5):
    """
    Creates demonstration outputs for the *bank* dataset.

    :param classifier: A method returning a trained classifier from X, y training pairs.
        Default is the `fit` method of sklearn's logistic regression for max_iter=1000.
    :param scaler: A method to preprocess features X. Default is the `fit_transform` of sklearn's `MinMaxScaler`.
    :param predict: Either "predict" (default) or "probabilities". The second option returns classification scores.
    :return: A tuple of the test set, desired binary labels, and predicted binary labels or their probabilities.
    """
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000).fit
    if scaler is None:
        scaler = MinMaxScaler().fit_transform
    data = read_csv(
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv",
        root=cache(),
        delimiter=";",
        header=0,
    )
    train, test = train_test_split(data, random_state=seed, test_size=test_size)
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
    if predict == "data":
        return x_train, y_train, x, y, train, test
    classifier = classifier(x_train, y_train)
    if predict == "predict":
        yhat = classifier.predict(x)
    elif predict == "probabilities":
        yhat = classifier.predict_proba(x)[:, 0]
    else:
        raise NotImplementedError()
    return test, y, yhat
