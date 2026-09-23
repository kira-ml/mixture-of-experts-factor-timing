"""Configuration loading and validation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL = ["run", "data", "features", "split", "models", "decision", "evaluation"]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)
    _validate(cfg)
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = hashlib.sha256(raw).hexdigest()
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    for k in REQUIRED_TOP_LEVEL:
        if k not in cfg:
            raise ValueError(f"Missing required config section: {k}")
    if not cfg["data"].get("universe"):
        raise ValueError("data.universe is empty")
    if not cfg["models"]:
        raise ValueError("models list is empty")