import numpy as np


def shrink_to_pd(cov: np.ndarray, alpha: float = 0.05, eps: float = 1e-6, max_eig: float = 1.0) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage toward scaled identity, with eigenvalue bounds."""
    n = cov.shape[0]
    cov = 0.5 * (cov + cov.T)
    mu = np.trace(cov) / n
    shrunk = (1.0 - alpha) * cov + alpha * mu * np.eye(n)
    eigvals, eigvecs = np.linalg.eigh(shrunk)
    eigvals = np.clip(eigvals, eps, max_eig)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T