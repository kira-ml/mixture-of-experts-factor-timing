"""Stage 8 — Evaluate. Metrics for predictive and decision layers."""
from __future__ import annotations

import numpy as np
from scipy import stats


def nll(y: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    N = len(y)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return float("inf")
    diff = y - mu
    sol = np.linalg.solve(L, diff)
    log_det = 2.0 * np.log(np.diag(L)).sum()
    return float(0.5 * (N * np.log(2 * np.pi) + log_det + sol @ sol))


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Per-asset CRPS under a diagonal Gaussian; averaged."""
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))))


def pit_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return stats.norm.cdf((y - mu) / sigma)


def annualized_return(net_returns: np.ndarray) -> float:
    return float((1 + net_returns).prod() ** (12 / len(net_returns)) - 1)


def annualized_vol(net_returns: np.ndarray) -> float:
    return float(np.std(net_returns, ddof=1) * np.sqrt(12))


def sharpe(net_returns: np.ndarray, rf: float = 0.0) -> float:
    excess = net_returns - rf / 12.0
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(12))


def sortino(net_returns: np.ndarray, rf: float = 0.0) -> float:
    excess = net_returns - rf / 12.0
    downside = excess[excess < 0]
    dd = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    if dd == 0:
        return 0.0
    return float(np.mean(excess) / dd * np.sqrt(12))


def max_drawdown(net_returns: np.ndarray) -> float:
    wealth = np.cumprod(1 + net_returns)
    peak = np.maximum.accumulate(wealth)
    dd = 1 - wealth / peak
    return float(dd.max())


def calmar(net_returns: np.ndarray) -> float:
    mdd = max_drawdown(net_returns)
    return float(annualized_return(net_returns) / mdd) if mdd > 0 else 0.0


def summary(net_returns: np.ndarray) -> dict:
    return {
        "ann_return": annualized_return(net_returns),
        "ann_vol": annualized_vol(net_returns),
        "sharpe": sharpe(net_returns),
        "sortino": sortino(net_returns),
        "max_drawdown": max_drawdown(net_returns),
        "calmar": calmar(net_returns),
        "mean_monthly": float(np.mean(net_returns)),
    }