"""Individual validation checks V1–V21 (subset enforced in MVP)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_all_dates_present(prices: pd.DataFrame) -> tuple[bool, str]:
    if prices.isna().all(axis=1).any():
        return False, "empty month rows present"
    return True, ""


def check_monotonic_dates(df: pd.DataFrame) -> tuple[bool, str]:
    if not df.index.is_monotonic_increasing:
        return False, "index not monotonic"
    if df.index.has_duplicates:
        return False, "duplicate dates"
    return True, ""


def check_numeric(df: pd.DataFrame) -> tuple[bool, str]:
    bad = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
    if bad:
        return False, f"non-numeric columns: {bad}"
    return True, ""


def check_finite(df: pd.DataFrame) -> tuple[bool, str]:
    arr = df.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        return False, f"non-finite values: {n_nan}"
    return True, ""


def check_nan_rate(df: pd.DataFrame, ceiling: float = 0.05) -> tuple[bool, str]:
    rates = df.isna().mean()
    bad = rates[rates > ceiling]
    if len(bad) > 0:
        return False, f"columns exceeding NaN rate {ceiling}: {bad.to_dict()}"
    return True, ""


def check_train_before_test(train_end: pd.Timestamp, test_start: pd.Timestamp) -> tuple[bool, str]:
    if not train_end < test_start:
        return False, f"train_end {train_end} not < test_start {test_start}"
    return True, ""