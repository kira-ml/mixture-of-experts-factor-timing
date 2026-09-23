"""I/O helpers: hashing, parquet round-trip, json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def save_parquet(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=True)
    return sha256_file(path)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def save_json(obj: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, indent=2, default=str).encode()
    path.write_bytes(raw)
    return sha256_bytes(raw)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())