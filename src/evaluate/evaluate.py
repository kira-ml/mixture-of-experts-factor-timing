"""Stage 8 orchestration — compute all metrics per model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest

from src.evaluate import bootstrap, metrics
from src.utils.io import save_json


def evaluate(cfg: dict, run_dir: Path, per_model: dict) -> dict:
    """
    per_model: {
      name: {
        "net_returns": np.ndarray,
        "turnover": np.ndarray,
        "costs": np.ndarray,
        "predictions": list of (y, mu, Sigma) or None for fixed models,
      }
    }
    """
    bootstrap_cfg = cfg["evaluation"]["bootstrap"]
    lam = cfg["decision"]["risk_aversion"]
    out: dict = {"models": {}}
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    for name, data in per_model.items():
        summary = metrics.summary(data["net_returns"])
        summary["total_turnover"] = float(np.sum(data["turnover"]))
        summary["expected_utility"] = metrics.expected_utility(data["net_returns"], lam)

        if data.get("predictions"):
            nlls, crps_vals, pit_vals = [], [], []
            for y, mu, Sigma in data["predictions"]:
                nlls.append(metrics.nll(y, mu, Sigma))
                sigma = np.sqrt(np.diag(Sigma))
                crps_vals.append(metrics.crps_gaussian(y, mu, sigma))
                pit_vals.append(metrics.pit_gaussian(y, mu, sigma))
            summary["nll_mean"] = float(np.mean(nlls))
            summary["crps_mean"] = float(np.mean(crps_vals))

            pit_flat = np.concatenate(pit_vals)
            ks_stat, ks_p = kstest(pit_flat, "uniform")
            summary["pit_ks_stat"] = float(ks_stat)
            summary["pit_ks_p"] = float(ks_p)
            np.save(run_dir / "metrics" / f"pit_{name}.npy", pit_flat)

        # Bootstrap CI on Sharpe and annual return
        if len(data["net_returns"]) >= 20:
            _, lo, hi = bootstrap.bootstrap_ci(
                data["net_returns"],
                metrics.sharpe,
                n_repetitions=bootstrap_cfg["n_repetitions"],
                block_length=bootstrap_cfg["block_length"],
                seed=bootstrap_cfg["seed"],
            )
            summary["sharpe_ci_lo"] = lo
            summary["sharpe_ci_hi"] = hi
            _, lo, hi = bootstrap.bootstrap_ci(
                data["net_returns"],
                lambda x: metrics.expected_utility(x, lam),
                n_repetitions=bootstrap_cfg["n_repetitions"],
                block_length=bootstrap_cfg["block_length"],
                seed=bootstrap_cfg["seed"],
            )
            summary["expected_utility_ci_lo"] = lo
            summary["expected_utility_ci_hi"] = hi

        out["models"][name] = summary

    save_json(out, run_dir / "metrics" / "summary.json")
    return out