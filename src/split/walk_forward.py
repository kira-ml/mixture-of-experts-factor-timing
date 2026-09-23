"""Stage 4 — Split. Expanding walk-forward train/test indices."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import load_parquet, save_json


def make_splits(cfg: dict, run_dir: Path) -> dict:
    feat_dir = run_dir / "features"
    X = load_parquet(feat_dir / "X.parquet")

    min_train = int(cfg["split"]["min_train"])
    step = int(cfg["split"]["test_step"])
    n = len(X)

    splits = []
    i = min_train
    while i < n:
        splits.append({
            "train_start_idx": 0,
            "train_end_idx": i - 1,
            "test_idx": i,
            "train_start": str(X.index[0].date()),
            "train_end": str(X.index[i - 1].date()),
            "test_date": str(X.index[i].date()),
        })
        i += step

    manifest = {"n_splits": len(splits), "splits": splits}
    save_json(manifest, run_dir / "splits" / "split_manifest.json")
    return manifest