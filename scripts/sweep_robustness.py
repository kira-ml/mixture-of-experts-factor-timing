"""Pre-committed robustness sweep over min_train and n_states.

Writes results/robustness_sweep.csv with one row per (min_train, n_states, model).
No changes to src/. Reuses the pipeline unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.run.orchestrator import run_pipeline
from src.utils.env import load_env


MIN_TRAIN_GRID = [60, 96, 132]
N_STATES_GRID = [2, 3]
BASE_CONFIG = "configs/default.yaml"


def main() -> None:
    load_env()
    base = load_config(BASE_CONFIG)

    rows = []
    for mt in MIN_TRAIN_GRID:
        for k in N_STATES_GRID:
            cfg = deepcopy(base)
            cfg["split"]["min_train"] = mt
            cfg["run"]["run_id"] = None
            for m in cfg["models"]:
                if m["name"] == "hmm_gaussian":
                    m["params"]["n_states"] = k
            print(f"\n===== min_train={mt}  n_states={k} =====")
            run_dir = run_pipeline(cfg)

            summary = pd.read_csv(run_dir / "report" / "summary.csv")
            avg_w = pd.read_csv(run_dir / "report" / "avg_weights.csv")

            for _, s in summary.iterrows():
                model = s["model"]
                cash = float(avg_w.loc[avg_w["model"] == model, "w_cash_residual"].iloc[0])
                rows.append({
                    "min_train": mt,
                    "n_states": k,
                    "model": model,
                    "sharpe": float(s["sharpe"]),
                    "expected_utility": float(s.get("expected_utility", float("nan"))),
                    "ann_return": float(s["ann_return"]),
                    "ann_vol": float(s["ann_vol"]),
                    "cash_residual": cash,
                    "nll_mean": s.get("nll_mean", None),
                    "pit_ks_p": s.get("pit_ks_p", None),
                    "run_id": run_dir.name,
                })

    df = pd.DataFrame(rows)
    out = Path("results") / "robustness_sweep.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[done] wrote {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()