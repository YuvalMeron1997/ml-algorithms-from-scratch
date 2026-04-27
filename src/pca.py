import numpy as np

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.std = None
        self.components = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=1)

        X_scaled = (X - self.mean) / self.std

        covariance_matrix = np.cov(X_scaled, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.mean) / self.std
        return X_scaled @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
