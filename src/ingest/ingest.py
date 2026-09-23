"""Stage 1 — Ingest. Pull raw data, save immutable snapshot, record provenance."""
from __future__ import annotations

from pathlib import Path

from src.ingest import fred_source, yfinance_source
from src.utils.io import save_json, save_parquet


def ingest(cfg: dict, run_dir: Path) -> dict:
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tickers = cfg["data"]["sources"]["yfinance"]["tickers"]
    start = cfg["data"]["start_date"]
    end = cfg["data"].get("end_date")

    prices = yfinance_source.fetch_monthly_prices(tickers, start, end)
    prices_hash = save_parquet(prices, raw_dir / "prices.parquet")

    macro_ids = cfg["data"]["sources"]["fred"]["series"]
    api_key_env = cfg["data"]["sources"]["fred"].get("api_key_env", "FRED_API_KEY")
    macro = fred_source.fetch_macro_series(macro_ids, api_key_env, start)
    macro_hash = save_parquet(macro, raw_dir / "macro.parquet")

    provenance = {
        "yfinance": {**yfinance_source.provenance_record(tickers, start, end), "hash": prices_hash},
        "fred": {**fred_source.provenance_record(macro_ids), "hash": macro_hash},
    }
    save_json(provenance, raw_dir / "provenance.json")
    return provenance