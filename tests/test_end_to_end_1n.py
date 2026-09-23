"""Smoke test: a run through the pipeline with only 1/N produces artifacts."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import load_config
from src.run.orchestrator import run_pipeline


@pytest.fixture
def minimal_cfg(tmp_path: Path) -> Path:
    cfg = {
        "run": {"run_id": "test_run", "seed": 42},
        "data": {
            "sources": {
                "yfinance": {"tickers": ["SPY", "IWD"], "frequency": "monthly"},
                "fred": {"series": [], "api_key_env": "FRED_API_KEY"},
            },
            "universe": ["SPY", "IWD"],
            "cash_proxy": "BIL",
            "start_date": "2015-01-01",
            "end_date": "2020-01-01",
        },
        "features": {
            "lagged_returns_lags": [1, 6, 12],
            "macro_series": [],
            "imputation": {"method": "ffill_only", "max_rate": 0.05},
        },
        "split": {"type": "expanding", "min_train": 24, "test_step": 1},
        "models": [{"name": "naive_1n", "type": "fixed", "method": "equal"}],
        "decision": {
            "utility": "mean_variance",
            "risk_aversion": 10,
            "constraints": {"long_only": True, "max_weight": 0.4, "turnover_cap": 0.5, "min_deployment": 0.0},
            "cost": {"mode": "assumed", "bps": 10.0},
            "solver": {"name": "cvxpy", "tolerance": 1e-6, "fallback_max_rate": 0.05},
        },
        "evaluation": {
            "primary_predictive_metric": "nll",
            "primary_decision_metric": "expected_utility",
            "bootstrap": {"method": "stationary_block", "block_length": 3, "n_repetitions": 50, "seed": 42},
            "deflated_sharpe": {"enabled": False},
        },
        "robustness": {"dimensions": []},
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.network
def test_end_to_end_1n(minimal_cfg: Path):
    cfg = load_config(minimal_cfg)
    run_dir = run_pipeline(cfg)
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "report" / "summary.csv").exists()