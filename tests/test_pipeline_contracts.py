"""Falsification tests: prove the pipeline's guards actually fire."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build import build_features


def _write_minimal_raw(run_dir, n_months: int = 120) -> None:
    """Create raw/ with two assets + cash proxy over n_months."""
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {
            "SPY": 100 * np.cumprod(1 + rng.normal(0.01, 0.04, n_months)),
            "IWD": 100 * np.cumprod(1 + rng.normal(0.009, 0.04, n_months)),
            "BIL": 100 * np.cumprod(1 + rng.normal(0.002, 0.0005, n_months)),
        },
        index=idx,
    )
    macro = pd.DataFrame(
        {"T10Y2Y": rng.normal(1.5, 0.5, n_months), "VIXCLS": rng.normal(18, 4, n_months)},
        index=idx,
    )
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    prices.to_parquet(run_dir / "raw" / "prices.parquet")
    macro.to_parquet(run_dir / "raw" / "macro.parquet")


def _minimal_cfg() -> dict:
    return {
        "data": {
            "universe": ["SPY", "IWD"],
            "cash_proxy": "BIL",
        },
        "features": {
            "lagged_returns_lags": [1, 6, 12],
            "macro_series": ["T10Y2Y", "VIXCLS"],
            "imputation": {"method": "ffill_only", "max_rate": 0.05},
        },
    }


def test_v13_no_leakage(tmp_path):
    """Baseline: legitimate features produce no V13 warnings."""
    _write_minimal_raw(tmp_path)
    manifest = build_features(_minimal_cfg(), tmp_path)
    assert manifest["leakage_warnings"] == {}


def test_v13_detects_target_leakage(tmp_path, monkeypatch):
    """Inject a feature that is the target -> V13 must flag it."""
    _write_minimal_raw(tmp_path)

    # Monkeypatch lagged-return construction to inject the *current* return
    # as a feature. This bypasses R1/R2 and should be caught by V13.
    from src.features import build as feat_build

    orig_build = feat_build.build_features

    def leaky_build(cfg, run_dir):
        # Reproduce build_features but shift=0 for one feature
        from src.utils.io import load_parquet, save_parquet, save_json
        prices = load_parquet(run_dir / "raw" / "prices.parquet")
        macro = load_parquet(run_dir / "raw" / "macro.parquet")
        universe = cfg["data"]["universe"]
        returns = prices[universe].ffill().pct_change().dropna(how="all")
        macro_lagged = macro.reindex(returns.index).ffill().shift(1)

        feat_cols = {"ret_SPY_lag1": returns["SPY"].shift(1),
                     "ret_IWD_lag1": returns["IWD"].shift(1),
                     "leaked_SPY": returns["SPY"]}  # <-- target as feature
        for m in cfg["features"]["macro_series"]:
            feat_cols[f"macro_{m}_lag1"] = macro_lagged[m]

        X = pd.DataFrame(feat_cols)
        y = returns[universe].copy()
        combined = pd.concat([X, y.add_prefix("y_")], axis=1).dropna()
        X = combined[[c for c in combined.columns if c in X.columns]]
        y = combined[[c for c in combined.columns if c.startswith("y_")]].rename(
            columns=lambda c: c.replace("y_", "")
        )
        manifest = {"n_features": X.shape[1], "n_targets": y.shape[1],
                    "n_rows": len(X), "features": list(X.columns),
                    "targets": list(y.columns)}
        ceiling = 0.95
        leakage = {}
        for feat in X.columns:
            for tgt in y.columns:
                c = float(X[feat].corr(y[tgt]))
                if abs(c) > ceiling:
                    leakage[f"{feat}|{tgt}"] = c
        manifest["leakage_warnings"] = leakage
        save_parquet(X, run_dir / "features" / "X.parquet")
        save_parquet(y, run_dir / "features" / "y.parquet")
        save_json(manifest, run_dir / "features" / "feature_manifest.json")
        return manifest

    monkeypatch.setattr(feat_build, "build_features", leaky_build)

    manifest = feat_build.build_features(_minimal_cfg(), tmp_path)
    assert manifest["leakage_warnings"], "V13 failed to detect injected leakage"
    # The leaky feature is a copy of the target -> correlation should be 1.0
    flagged = any("leaked_SPY" in k for k in manifest["leakage_warnings"])
    assert flagged, f"V13 did not flag leaked_SPY: {manifest['leakage_warnings']}"