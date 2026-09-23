"""CLI entry point: python -m src.run --config configs/default.yaml."""
from __future__ import annotations

import argparse

from src.config import load_config
from src.run.orchestrator import run_pipeline
from src.utils.env import load_env


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()