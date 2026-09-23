"""Stage 7 — Backtest. Walk-forward portfolio simulation with costs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(
    weights: np.ndarray,
    realized: np.ndarray,
    w_prev: np.ndarray,
    cost_bps: float,
    cash_return: float = 0.0,
) -> dict:
    cash_weight = max(0.0, 1.0 - float(weights.sum()))
    gross = float(weights @ realized + cash_weight * cash_return)
    turnover = 0.5 * float(np.abs(weights - w_prev).sum())
    cost = float(cost_bps / 10000.0 * np.abs(weights - w_prev).sum())
    net = gross - cost
    return {"gross_return": gross, "net_return": net, "turnover": turnover, "cost": cost}