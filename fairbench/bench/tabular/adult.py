from fairbench.bench.loader import read_csv, features
from fairbench.fallbacks import LogisticRegression, MinMaxScaler, train_test_split
import numpy as np
from fairbench.bench.loader import cache


def adult(
    classifier=None,
    scaler=None,
    predict="predict",
):
    """
    Creates demonstration outputs for the *adult* dataset.

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
    train = read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        root=cache(),
        header=None,
        skipinitialspace=True,
    )
    test = read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        root=cache(),
        header=None,
        skipinitialspace=True,
        skiprows=[0],
    )
    numeric = [0, 4, 11, 12]
    categorical = [
        3,
        5,
        6,
        8,
        9,
    ]  # column 1 is also categorical but yields different get_dummies between train and test
    x_train = features(train, numeric, categorical)
    y_train = (train[14] == ">50K").values
    x = features(test, numeric, categorical)
    y = (test[14] == ">50K.").values

    # Apply scaler and replace None values with zero
    x_train = np.nan_to_num(scaler(x_train), nan=0.0)
    x = np.nan_to_num(scaler(x), nan=0.0)

    if predict == "data":
        return x_train, y_train, x, y, train, test
    classifier = classifier(x_train, y_train)
    if predict == "predict":
        yhat = classifier.predict(x)
    elif predict == "probabilities":
        yhat = classifier.predict_proba(x)[:, 1]
    else:
        raise NotImplementedError()
    return test, y, yhat
