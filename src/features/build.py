"""Stage 3 — Features. Build 17-feature matrix and target from validated snapshot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import load_parquet, save_json, save_parquet


def build_features(cfg: dict, run_dir: Path) -> dict:
    raw = run_dir / "raw"
    prices = load_parquet(raw / "prices.parquet")
    macro = load_parquet(raw / "macro.parquet")

    universe: list[str] = cfg["data"]["universe"]
    cash_proxy: str = cfg["data"]["cash_proxy"]
    lags: list[int] = cfg["features"]["lagged_returns_lags"]
    macro_ids: list[str] = cfg["features"]["macro_series"]

    # Monthly returns for the investable universe
    px = prices[universe].ffill()
    returns = px.pct_change().dropna(how="all")

    # Align macro to the same monthly index, then lag 1 month (conservative)
    macro_aligned = macro.reindex(returns.index).ffill()
    macro_lagged = macro_aligned.shift(1)

    feat_cols: dict[str, pd.Series] = {}

    # Lagged returns: 5 assets × len(lags) = 15 features
    for asset in universe:
        for L in lags:
            col = f"ret_{asset}_lag{L}"
            feat_cols[col] = returns[asset].shift(L)

    # Macro features: 2 features, already lagged
    for m in macro_ids:
        feat_cols[f"macro_{m}_lag1"] = macro_lagged[m]

    X = pd.DataFrame(feat_cols)
    y = returns[universe].copy()

    # Align
    combined = pd.concat([X, y.add_prefix("y_")], axis=1).dropna()
    X = combined[[c for c in combined.columns if c in X.columns]]
    y = combined[[c for c in combined.columns if c.startswith("y_")]].rename(
        columns=lambda c: c.replace("y_", "")
    )

    # Feature manifest
    manifest = {
        "n_features": int(X.shape[1]),
        "n_targets": int(y.shape[1]),
        "n_rows": int(len(X)),
        "features": [
            {"name": c, "category": ("macro" if c.startswith("macro_") else "lagged_return")}
            for c in X.columns
        ],
        "targets": list(y.columns),
        "T_over_p": float(len(X) / max(X.shape[1], 1)),
    }

    # V13 — target leakage correlation check (warn only, per 06 Section 9.3)
    ceiling = 0.95
    leakage: dict[str, float] = {}
    for feat in X.columns:
        for tgt in y.columns:
            c = float(X[feat].corr(y[tgt]))
            if abs(c) > ceiling:
                leakage[f"{feat}|{tgt}"] = c
    manifest["leakage_warnings"] = leakage
    if leakage:
        print(f"[V13] potential leakage (>{ceiling}): {leakage}")

    feat_dir = run_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(X, feat_dir / "X.parquet")
    save_parquet(y, feat_dir / "y.parquet")
    save_json(manifest, feat_dir / "feature_manifest.json")
    return manifest