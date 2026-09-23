"""Fixed allocation strategies (non-optimized)."""
from __future__ import annotations

import numpy as np


class EqualWeight:
    name = "naive_1n"

    def __init__(self, n_assets: int):
        self.n = n_assets

    def weights(self, t_idx: int, returns: np.ndarray) -> np.ndarray:
        return np.full(self.n, 1.0 / self.n)


class CashOnly:
    name = "cash"

    def __init__(self, n_assets: int):
        self.n = n_assets

    def weights(self, t_idx: int, returns: np.ndarray) -> np.ndarray:
        return np.zeros(self.n)


class Persistence:
    name = "persistence"

    def __init__(self, n_assets: int):
        self.n = n_assets

    def weights(self, t_idx: int, returns: np.ndarray) -> np.ndarray:
        if t_idx < 1:
            return np.full(self.n, 1.0 / self.n)
        prev = returns[t_idx - 1]
        return _long_positive_weights(prev)


class RollingMean:
    name = "rolling_avg"

    def __init__(self, n_assets: int, window: int = 12):
        self.n = n_assets
        self.window = window

    def weights(self, t_idx: int, returns: np.ndarray) -> np.ndarray:
        if t_idx < self.window:
            return np.full(self.n, 1.0 / self.n)
        window = returns[t_idx - self.window : t_idx]
        mu = window.mean(axis=0)
        return _long_positive_weights(mu)


class Momentum:
    name = "momentum"

    def __init__(self, n_assets: int, decay: float = 0.9, window: int = 12):
        self.n = n_assets
        self.decay = decay
        self.window = window

    def weights(self, t_idx: int, returns: np.ndarray) -> np.ndarray:
        if t_idx < self.window:
            return np.full(self.n, 1.0 / self.n)
        window = returns[t_idx - self.window : t_idx]
        w = np.array([self.decay ** (self.window - 1 - i) for i in range(self.window)])
        w = w / w.sum()
        mu = (window * w[:, None]).sum(axis=0)
        return _long_positive_weights(mu)


def _long_positive_weights(signal: np.ndarray) -> np.ndarray:
    """Project a signal into the simplex: long-only, sums to 1, cap at 0.40."""
    s = np.clip(signal, 0, None)
    if s.sum() <= 0:
        return np.full(len(s), 1.0 / len(s))
    w = s / s.sum()
    # Simple cap-and-renormalize loop (few assets, converges quickly)
    for _ in range(20):
        over = w > 0.40
        if not over.any():
            break
        excess = (w[over] - 0.40).sum()
        w[over] = 0.40
        free = ~over
        if free.any():
            w[free] += excess * (w[free] / w[free].sum()) if w[free].sum() > 0 else excess / free.sum()
    return w / w.sum()