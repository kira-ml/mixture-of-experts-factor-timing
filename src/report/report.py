"""Stage 9 — Report. CSV summary + basic figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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