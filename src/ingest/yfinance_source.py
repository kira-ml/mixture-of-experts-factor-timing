"""Pull monthly prices from yfinance."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


def fetch_monthly_prices(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Return monthly adjusted close prices, index = month-end, columns = tickers."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty frame")

    # Handle multi-ticker vs single-ticker shapes
    if isinstance(raw.columns, pd.MultiIndex):
        adj = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)})
    else:
        adj = raw[["Close"]].rename(columns={"Close": tickers[0]})

    adj = adj.sort_index()
    monthly = adj.resample("ME").last().dropna(how="all")
    return monthly


def provenance_record(tickers: list[str], start: str, end: str | None) -> dict:
    return {
        "source": "yfinance",
        "tickers": tickers,
        "start": start,
        "end": end,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }