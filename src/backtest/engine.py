"""Stage 7 — Backtest. Walk-forward portfolio simulation with costs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(
    weights: np.ndarray,
    realized: np.ndarray,
    w_prev: np.ndarray,
    cost_bps: float,
) -> dict:
    gross = float(weights @ realized)
    turnover = 0.5 * float(np.abs(weights - w_prev).sum())
    cost = float(cost_bps / 10000.0 * np.abs(weights - w_prev).sum())
    net = gross - cost
    return {"gross_return": gross, "net_return": net, "turnover": turnover, "cost": cost}