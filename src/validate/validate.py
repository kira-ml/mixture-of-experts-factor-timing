"""Stage 2 — Validate. Enforce data contract; abort on failure."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import load_parquet, save_json
from src.validate import checks


def validate(cfg: dict, run_dir: Path) -> dict:
    raw_dir = run_dir / "raw"
    prices = load_parquet(raw_dir / "prices.parquet")
    macro = load_parquet(raw_dir / "macro.parquet")

    universe = cfg["data"]["universe"]
    cash_proxy = cfg["data"]["cash_proxy"]

    report: dict = {"checks": {}, "aborted": False}

    def run_check(vid: str, fn, *args):
        ok, msg = fn(*args)
        report["checks"][vid] = {"pass": ok, "message": msg}
        if not ok:
            report["aborted"] = True

    run_check("V1", checks.check_all_dates_present, prices)
    run_check("V3", checks.check_monotonic_dates, prices)
    run_check("V3b", checks.check_monotonic_dates, macro)
    run_check("V4", checks.check_numeric, prices)
    run_check("V4b", checks.check_numeric, macro)

    missing = [t for t in universe + [cash_proxy] if t not in prices.columns]
    report["checks"]["V1b"] = {"pass": len(missing) == 0, "message": f"missing tickers: {missing}"}
    if missing:
        report["aborted"] = True

    # Drop cash proxy from feature frame; it's used for risk-free rate only
    prices_no_cash = prices.drop(columns=[cash_proxy], errors="ignore")
    filled = prices_no_cash.ffill()
    run_check("V5", checks.check_finite, filled)
    run_check("V2", checks.check_nan_rate, filled, cfg["features"]["imputation"]["max_rate"])

    save_json(report, run_dir / "validation_report.json")
    if report["aborted"]:
        raise RuntimeError(f"Validation aborted: {report}")
    return report