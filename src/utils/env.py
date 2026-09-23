"""Load .env once, at process start."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_env(repo_root: Path | None = None) -> None:
    """Load .env from the repo root. Silently no-ops if absent."""
    root = repo_root or Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)