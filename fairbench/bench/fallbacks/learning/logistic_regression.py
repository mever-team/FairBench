import numpy as np


class LogisticRegression:
    def __init__(
        self,
        learning_rate=0.1,
        max_iter=1000,
        tol=1e-6,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Adam parameters
        m_w, v_w = np.zeros(num_features), np.zeros(
            num_features
        )  # Moment vectors for weights
        m_b, v_b = 0, 0  # Moment vectors for bias
        t = 0  # Time step

        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            if np.sum(np.abs(dw)) < self.tol:
                break

            t += 1  # Increment time step

            # Update weights with Adam
            m_w = self.beta1 * m_w + (1 - self.beta1) * dw
            v_w = self.beta2 * v_w + (1 - self.beta2) * (dw**2)
            m_w_hat = m_w / (1 - self.beta1**t)
            v_w_hat = v_w / (1 - self.beta2**t)
            self.weights -= (
                self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            )

            # Update bias with Adam
            m_b = self.beta1 * m_b + (1 - self.beta1) * db
            v_b = self.beta2 * v_b + (1 - self.beta2) * (db**2)
            m_b_hat = m_b / (1 - self.beta1**t)
            v_b_hat = v_b / (1 - self.beta2**t)
            self.bias -= (
                self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            )

        return self

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        prediction = self.sigmoid(linear_model)
        ret = np.column_stack([1 - prediction, prediction])
        return ret

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        prediction = self.sigmoid(linear_model)
        return prediction > 0.5
