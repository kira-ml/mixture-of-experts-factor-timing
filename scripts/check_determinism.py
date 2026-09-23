from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from pathlib import Path

from src.config import load_config
from src.run.orchestrator import run_pipeline
from src.utils.env import load_env
from src.utils.io import load_json


def run_once(base_cfg: dict, run_id: str) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["run"]["run_id"] = run_id
    run_dir = run_pipeline(cfg)
    return load_json(run_dir / "manifest.json")


def main() -> int:
    load_env()

    # Clean up any prior pass dirs (orchestrator refuses to overwrite)
    for rid in ("detcheck_pass1", "detcheck_pass2"):
        p = Path("runs") / rid
        if p.exists():
            shutil.rmtree(p)

    base = load_config("configs/default.yaml")
    # Keep the run small so the check is fast
    base["split"]["min_train"] = 60
    base["data"]["end_date"] = "2023-01-01"
    # Remove the sweep dimensions and reduce models to speed up
    base["models"] = [m for m in base["models"] if m["name"] in
                      {"naive_1n", "ridge", "hmm_gaussian"}]

    print("Running pass 1 ...")
    m1 = run_once(base, "detcheck_pass1")
    print("Running pass 2 ...")
    m2 = run_once(base, "detcheck_pass2")

    # Provenance records the retrieval UTC timestamp, which by design differs
    # every run. It is metadata, not a result artifact (08 Section 7.1).
    IGNORE = {str(Path("raw") / "provenance.json")}
    a1, a2 = m1.get("artifacts", {}), m2.get("artifacts", {})
    keys = sorted(k for k in (set(a1) | set(a2)) if k not in IGNORE)
    diffs = [k for k in keys if a1.get(k) != a2.get(k)]

    if not diffs:
        print(f"\nOK: {len(keys)} artifacts identical across two runs.")
        return 0
    print(f"\nFAIL: {len(diffs)}/{len(keys)} artifacts differ.")
    for k in diffs:
        print(f"  {k}\n    pass1: {a1.get(k)}\n    pass2: {a2.get(k)}")
    return 1

if __name__ == "__main__":
    sys.exit(main())