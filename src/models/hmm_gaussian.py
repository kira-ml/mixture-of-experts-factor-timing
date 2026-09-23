"""Gaussian HMM with mixture mean/covariance predictive distribution."""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMGaussianModel:
    name = "hmm_gaussian"

    def __init__(self, n_states: int = 2, covariance_type: str = "diag", n_iter: int = 100, seed: int = 42):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.seed = seed
        self.model: GaussianHMM | None = None
        self.X_train: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # HMM is fit on the target series directly (returns)
        # Features are used by the caller only to select the training window.
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.seed,
        )
        self.model.fit(y_train)
        self.X_train = y_train

    def predict_distribution(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None and self.X_train is not None
        # Use forward-backward to get posterior over states at the last training point
        _, posteriors = self.model.score_samples(self.X_train)
        pi = posteriors[-1]  # (K,)
        mu_k = self.model.means_  # (K, N)
        covs = self.model.covars_
        if self.covariance_type == "diag":
            if covs.ndim == 3:
                # hmmlearn 0.3.x returns (K, N, N) diagonal matrices for 'diag'
                cov_k = covs
            else:
                # older hmmlearn returned (K, N) variances
                cov_k = np.array([np.diag(c) for c in covs])
        else:
            cov_k = covs
        mu = (pi[:, None] * mu_k).sum(axis=0)
        diff = mu_k - mu[None, :]
        Sigma = np.zeros((mu.shape[0], mu.shape[0]))
        for k in range(self.n_states):
            Sigma += pi[k] * (cov_k[k] + np.outer(diff[k], diff[k]))
        return mu, _shrink_to_pd(Sigma)


def _shrink_to_pd(cov: np.ndarray, alpha: float = 0.05, eps: float = 1e-6, max_eig: float = 1.0) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage toward scaled identity, with eigenvalue bounds."""
    n = cov.shape[0]
    cov = 0.5 * (cov + cov.T)
    mu = np.trace(cov) / n
    shrunk = (1.0 - alpha) * cov + alpha * mu * np.eye(n)
    eigvals, eigvecs = np.linalg.eigh(shrunk)
    eigvals = np.clip(eigvals, eps, max_eig)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T