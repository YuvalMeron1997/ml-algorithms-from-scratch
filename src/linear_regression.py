import numpy as np

class LinearRegressionFromScratch:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        X_bias = np.c_[np.ones(X.shape[0]), X]
        coefficients = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

        self.bias = coefficients[0]
        self.weights = coefficients[1:]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias
