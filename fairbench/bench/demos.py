from fairbench.bench.loader import read_csv, features
import sklearn

import sklearn.linear_model
import sklearn.preprocessing
import numpy as np


def adult(
    classifier=sklearn.linear_model.LogisticRegression(max_iter=1000).fit,
    scaler=sklearn.preprocessing.MinMaxScaler().fit_transform,
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


def bank(
    classifier=sklearn.linear_model.LogisticRegression(max_iter=1000).fit,
    scaler=sklearn.preprocessing.MinMaxScaler().fit_transform,
    predict="predict",
    seed=None,
):
    """
    Creates demonstration outputs for the *bank* dataset.

    :param classifier: A method returning a trained classifier from X, y training pairs.
        Default is the `fit` method of sklearn's logistic regression for max_iter=1000.
    :param scaler: A method to preprocess features X. Default is the `fit_transform` of sklearn's `MinMaxScaler`.
    :param predict: Either "predict" (default) or "probabilities". The second option returns classification scores.
    :return: A tuple of the test set, desired binary labels, and predicted binary labels or their probabilities.
    """
    data = read_csv(
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv",
        delimiter=";",
    )
    train, test = sklearn.model_selection.train_test_split(data, random_state=seed)
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


def compas(
    classifier=sklearn.linear_model.LogisticRegression(max_iter=1000).fit,
    scaler=sklearn.preprocessing.MinMaxScaler().fit_transform,
    predict="predict",
    seed=None,
):
    """
    Creates demonstration outputs for the *compas* dataset.

    :param classifier: A method returning a trained classifier from X, y training pairs.
        Default is the `fit` method of sklearn's logistic regression for max_iter=1000.
    :param scaler: A method to preprocess features X. Default is the `fit_transform` of sklearn's `MinMaxScaler`.
    :param predict: Either "predict" (default) or "probabilities". The second option returns classification scores.
    :return: A tuple of the test set, desired binary labels, and predicted binary labels or their probabilities.
    """
    data = read_csv(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        delimiter=",",
    )
    train, test = sklearn.model_selection.train_test_split(data, random_state=seed)
    numeric = [
        "age",
        # "juv_fel_count",
        # "decile_score",
        # "juv_misd_count",
        # "juv_other_count",
        # "priors_count",
        # "days_b_screening_arrest",
    ]
    categorical = [
        "sex",
        "c_charge_degree",
        "race",
        "is_recid",
        # "violent_recid",
        # "is_violent_recid",
        # "vr_charge_degree",
        # "v_decile_score"
    ]
    x_train = features(train, numeric, categorical)
    y_train = (train["two_year_recid"] == 1).values
    x = features(test, numeric, categorical)
    y = (test["two_year_recid"] == 1).values
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
