"""Stage 10 — Orchestrator. Runs Stages 1–9 and writes manifest."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import engine
from src.decision import mean_variance
from src.evaluate import evaluate as eval_stage
from src.features import build as features_stage
from src.ingest import ingest as ingest_stage
from src.models import fixed as fixed_models
from src.models.hmm_gaussian import HMMGaussianModel
from src.models.ridge import RidgeModel
from src.report import report as report_stage
from src.run import manifest as manifest_stage
from src.split import walk_forward
from src.utils.io import load_parquet
from src.utils.seeds import set_global_seed
from src.validate import validate as validate_stage


def run_pipeline(cfg: dict) -> Path:
    set_global_seed(cfg["run"]["seed"])
    run_id = cfg["run"]["run_id"] or _make_run_id()
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[run] {run_id}")

    print("[1] ingest")
    provenance = ingest_stage.ingest(cfg, run_dir)

    print("[2] validate")
    validation = validate_stage.validate(cfg, run_dir)

    print("[3] features")
    features_stage.build_features(cfg, run_dir)

    print("[4] split")
    splits = walk_forward.make_splits(cfg, run_dir)

    print("[5-7] models / decision / backtest")
    per_model = _run_models(cfg, run_dir, splits)

    print("[8] evaluate")
    eval_summary = eval_stage.evaluate(cfg, run_dir, per_model)

    print("[9] report")
    report_stage.report(cfg, run_dir, per_model, eval_summary)

    print("[10] manifest")
    manifest_stage.write_manifest(cfg, run_dir, per_model, validation, provenance)

    print(f"[done] {run_dir}")
    return run_dir


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_models(cfg: dict, run_dir: Path, splits: dict) -> dict:
    X = load_parquet(run_dir / "features" / "X.parquet")
    y = load_parquet(run_dir / "features" / "y.parquet")
    prices = load_parquet(run_dir / "raw" / "prices.parquet")

    universe = cfg["data"]["universe"]
    cash_proxy = cfg["data"]["cash_proxy"]
    n = len(universe)

    cost_bps = cfg["decision"]["cost"]["bps"]
    lam = cfg["decision"]["risk_aversion"]
    constraints = cfg["decision"]["constraints"]

    X_np = X.to_numpy()
    y_np = y.to_numpy()
    returns_universe = prices[universe].pct_change().reindex(X.index).fillna(0.0).to_numpy()
    cash_returns = prices[cash_proxy].pct_change().reindex(X.index).fillna(0.0).to_numpy()

    results: dict[str, dict] = {}

    for mcfg in cfg["models"]:
        name = mcfg["name"]
        print(f"  model: {name}")
        mtype = mcfg["type"]
        net_returns, turnovers, costs = [], [], []
        weights_list = []
        predictions = [] if mtype != "fixed" else None

        w_prev = np.zeros(n)  # start in cash
        allocator = None
        probabilistic = None

        if mtype == "fixed":
            allocator = _build_fixed(mcfg, n)
        elif name == "ridge":
            probabilistic = RidgeModel(alpha=mcfg["params"]["alpha"], seed=cfg["run"]["seed"])
        elif name == "hmm_gaussian":
            p = mcfg["params"]
            probabilistic = HMMGaussianModel(
                n_states=p["n_states"],
                covariance_type=p["covariance_type"],
                n_iter=p["n_iter"],
                seed=cfg["run"]["seed"],
            )
        else:
            raise ValueError(f"Unknown model: {name}")

        for split in splits["splits"]:
            t_train_end = split["train_end_idx"]
            t_test = split["test_idx"]

            X_train = X_np[: t_train_end + 1]
            y_train = y_np[: t_train_end + 1]
            X_test = X_np[t_test]

            if mtype == "fixed":
                w = allocator.weights(t_test, returns_universe)
            else:
                probabilistic.fit(X_train, y_train)
                mu, Sigma = probabilistic.predict_distribution(X_test)
                w, _status = mean_variance.solve(
                    mu=mu,
                    Sigma=Sigma,
                    w_prev=w_prev,
                    risk_aversion=lam,
                    cost_bps=cost_bps,
                    max_weight=constraints["max_weight"],
                    turnover_cap=constraints["turnover_cap"],
                )
                predictions.append((y_np[t_test], mu, Sigma))

            realized = returns_universe[t_test]
            metrics_t = engine.compute_returns(w, realized, w_prev, cost_bps, cash_return=float(cash_returns[t_test]))
            net_returns.append(metrics_t["net_return"])
            turnovers.append(metrics_t["turnover"])
            costs.append(metrics_t["cost"])
            weights_list.append(w.copy())
            w_prev = w

        results[name] = {
            "net_returns": np.array(net_returns),
            "turnover": np.array(turnovers),
            "costs": np.array(costs),
            "predictions": predictions,
            "cash_returns": cash_returns[len(cash_returns) - len(net_returns):],
            "weights": np.array(weights_list),
        }

        weights_df = pd.DataFrame(
            weights_list,
            index=[s["test_date"] for s in splits["splits"]],
            columns=universe,
        )
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_df.to_parquet(weights_dir / f"{name}.parquet")

    return results


def _build_fixed(mcfg: dict, n: int):
    method = mcfg["method"]
    params = mcfg.get("params", {})
    if method == "equal":
        return fixed_models.EqualWeight(n)
    if method == "cash":
        return fixed_models.CashOnly(n)
    if method == "persistence":
        return fixed_models.Persistence(n)
    if method == "rolling_mean":
        return fixed_models.RollingMean(n, window=params.get("window", 12))
    if method == "momentum":
        return fixed_models.Momentum(n, decay=params.get("decay", 0.9), window=params.get("window", 12))
    raise ValueError(f"Unknown fixed method: {method}")