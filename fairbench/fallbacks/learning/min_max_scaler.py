import numpy as np


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.feature_range = feature_range

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X):
        X_std = (X - self.min) / (self.max - self.min)
        X_scaled = (
            X_std * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
