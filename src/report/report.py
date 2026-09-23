"""Stage 9 — Report. CSV summary + basic figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils.io import save_json


def report(cfg: dict, run_dir: Path, per_model: dict, eval_summary: dict) -> None:
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    rows = []
    for name, s in eval_summary["models"].items():
        rows.append({"model": name, **s})
    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "summary.csv", index=False)

    # Cumulative wealth plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in per_model.items():
        wealth = (1 + data["net_returns"]).cumprod()
        ax.plot(wealth, label=name)
    ax.set_title("Cumulative net wealth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "cumulative_wealth.png", dpi=120)
    plt.close(fig)



    # Average weights per model — diagnostic for cash residual / exposure
    weights_dir = run_dir / "weights"
    if weights_dir.exists():
        rows = []
        for f in sorted(weights_dir.glob("*.parquet")):
            w = pd.read_parquet(f)
            row = {"model": f.stem}
            mean_w = w.mean(axis=0)
            for col in w.columns:
                row[f"w_{col}"] = float(mean_w[col])
            row["w_cash_residual"] = float(1.0 - mean_w.sum())
            rows.append(row)
        pd.DataFrame(rows).to_csv(report_dir / "avg_weights.csv", index=False)


    # PIT histograms for probabilistic models
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        pit_files = sorted(metrics_dir.glob("pit_*.npy"))
        if pit_files:
            fig, axes = plt.subplots(1, len(pit_files), figsize=(6 * len(pit_files), 4), squeeze=False)
            for ax, f in zip(axes[0], pit_files):
                u = np.load(f)
                ax.hist(u, bins=20, range=(0, 1), density=True, edgecolor="black")
                ax.axhline(1.0, color="red", linestyle="--")
                ax.set_title(f"PIT — {f.stem.replace('pit_', '')}")
                ax.set_xlabel("u = F(y)")
                ax.set_ylabel("density")
            fig.tight_layout()
            fig.savefig(report_dir / "pit_histograms.png", dpi=120)
            plt.close(fig)


            
    # Disclosures
    disclosures = {
        "non_goals": [
            "This project is not a trading strategy.",
            "This project does not claim to beat any benchmark.",
            "This project does not claim that regimes exist in the market.",
            "This project does not claim that backtest results imply live performance.",
            "This project does not claim statistical significance without tests.",
        ]
    }
    save_json(disclosures, report_dir / "disclosures.json")