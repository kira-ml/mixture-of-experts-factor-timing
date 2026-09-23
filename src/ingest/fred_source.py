"""Pull macro series from FRED."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
from fredapi import Fred


def fetch_macro_series(series_ids: list[str], api_key_env: str, start: str) -> pd.DataFrame:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set")
    fred = Fred(api_key=api_key)
    frames = {}
    for sid in series_ids:
        s = fred.get_series(sid, observation_start=start)
        s.name = sid
        frames[sid] = s
    df = pd.concat(frames.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    monthly = df.resample("ME").last()
    return monthly


def provenance_record(series_ids: list[str]) -> dict:
    return {
        "source": "fred",
        "series": series_ids,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "vintage_policy": "latest_release_minus_one_month_lag",
    }