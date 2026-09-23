"""One-off lambda sensitivity sweep. Reuses the existing pipeline unchanged.

Runs the current config for each lambda in the grid, then prints a summary
table: lambda × model × {sharpe, cash_residual, ann_return}.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.run.orchestrator import run_pipeline
from src.utils.env import load_env


GRID = [2, 5, 10, 20]
BASE_CONFIG = "configs/default.yaml"
BASE = load_config(BASE_CONFIG)


def main() -> None:
    load_env()

    rows = []
    for lam in GRID:
        cfg = deepcopy(BASE)
        cfg["decision"]["risk_aversion"] = lam
        cfg["run"]["run_id"] = None  # auto-generate
        print(f"\n===== lambda = {lam} =====")
        run_dir = run_pipeline(cfg)

        summary = pd.read_csv(run_dir / "report" / "summary.csv")
        avg_w = pd.read_csv(run_dir / "report" / "avg_weights.csv")

        for _, s in summary.iterrows():
            model = s["model"]
            cash = float(avg_w.loc[avg_w["model"] == model, "w_cash_residual"].iloc[0])
            rows.append({
                "lambda": lam,
                "model": model,
                "sharpe": float(s["sharpe"]),
                "ann_return": float(s["ann_return"]),
                "ann_vol": float(s["ann_vol"]),
                "cash_residual": cash,
                "nll_mean": s.get("nll_mean", None),
                "run_id": run_dir.name,
            })

    df = pd.DataFrame(rows)
    out = Path("results") / "lambda_sweep.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[done] wrote {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()