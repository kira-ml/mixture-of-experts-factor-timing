"""Stationary block bootstrap for confidence intervals."""
from __future__ import annotations

import numpy as np


def stationary_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / block_length
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        length = rng.geometric(p)
        for j in range(length):
            if i >= n:
                break
            idx[i] = (start + j) % n
            i += 1
    return idx


def bootstrap_ci(
    series: np.ndarray,
    statistic_fn,
    n_repetitions: int = 1000,
    block_length: int = 3,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    stats = np.empty(n_repetitions)
    n = len(series)
    for r in range(n_repetitions):
        idx = stationary_bootstrap_indices(n, block_length, rng)
        stats[r] = statistic_fn(series[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(stats)), float(lo), float(hi)