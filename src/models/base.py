"""Model interfaces."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class FixedAllocator(Protocol):
    """Maps history to a weight vector over the universe at time t. Cash is residual."""
    def weights(self, t_idx: int, returns: np.ndarray) -> np.ndarray: ...


class ProbabilisticModel(Protocol):
    """Fits on train, produces μ and Σ on test."""
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None: ...
    def predict_distribution(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...