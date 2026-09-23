"""Constrained mean-variance decision rule via cvxpy."""
from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


def solve(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    risk_aversion: float,
    cost_bps: float,
    max_weight: float,
    turnover_cap: float,
) -> tuple[np.ndarray, str]:
    """Return (weights, solver_status). Weights sum to <= 1; cash is residual."""
    n = mu.shape[0]
    c = cost_bps / 10000.0

    if not _HAS_CVXPY:
        return _fallback(w_prev), "fallback_no_cvxpy"

    w = cp.Variable(n)
    objective = cp.Maximize(
        w @ mu - (risk_aversion / 2.0) * cp.quad_form(w, cp.psd_wrap(Sigma)) - c * cp.norm1(w - w_prev)
    )
    constraints = [
        w >= 0,
        cp.sum(w) <= 1.0,
        w <= max_weight,
        cp.norm1(w - w_prev) <= turnover_cap,
    ]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        try:
            prob.solve(verbose=False)
        except Exception:
            return _fallback(w_prev), "fallback_solver_exception"
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return _fallback(w_prev), f"fallback_{prob.status}"
    w_val = np.clip(w.value, 0.0, None)
    if w_val.sum() > 1.0:
        w_val = w_val / w_val.sum()
    return w_val, "optimal"


def _fallback(w_prev: np.ndarray) -> np.ndarray:
    return w_prev.copy()