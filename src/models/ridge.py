"""Ridge regression with Gaussian residual covariance."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class RidgeModel:
    name = "ridge"

    def __init__(self, alpha: float = 1.0, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.model: Ridge | None = None
        self.resid_cov: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = Ridge(alpha=self.alpha, random_state=self.seed)
        self.model.fit(X_train, y_train)
        resid = y_train - self.model.predict(X_train)
        # Ledoit-Wolf-style shrinkage to make covariance PD
        cov = np.cov(resid, rowvar=False)
        cov = _shrink_to_pd(cov)
        self.resid_cov = cov

    def predict_distribution(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None and self.resid_cov is not None
        mu = self.model.predict(X_test.reshape(1, -1))[0]
        return mu, self.resid_cov


def _shrink_to_pd(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T