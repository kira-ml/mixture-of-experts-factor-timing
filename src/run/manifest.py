"""Run manifest writer."""
from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import save_json, sha256_file


def write_manifest(
    cfg: dict,
    run_dir: Path,
    per_model: dict,
    validation: dict,
    ingest_provenance: dict,
) -> dict:
    manifest = {
        "run_id": run_dir.name,
        "config_path": cfg["_config_path"],
        "config_hash": cfg["_config_hash"],
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "seeds": {"global": cfg["run"]["seed"]},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "ingest_provenance": ingest_provenance,
        "validation_aborted": validation.get("aborted", False),
        "models_run": list(per_model.keys()),
        "artifacts": {},
    }
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            rel = str(p.relative_to(run_dir))
            manifest["artifacts"][rel] = sha256_file(p)
    save_json(manifest, run_dir / "manifest.json")
    return manifest