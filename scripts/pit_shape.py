"""One-off PIT shape diagnostic. Reads saved npy, no pipeline re-run."""
from __future__ import annotations

from pathlib import Path

import numpy as np

RUN = Path("runs") / "20260923T162919Z"

for f in sorted((RUN / "metrics").glob("pit_*.npy")):
    u = np.load(f)
    print(f"\n{f.stem}")
    print(f"  n = {len(u)}")
    print(f"  quantiles (5, 25, 50, 75, 95): "
          f"{np.quantile(u, [0.05, 0.25, 0.5, 0.75, 0.95])}")
    print(f"  mean = {u.mean():.4f}          (uniform: 0.5000)")
    print(f"  frac < 0.05 = {(u < 0.05).mean():.4f}  (uniform: 0.0500)")
    print(f"  frac > 0.95 = {(u > 0.95).mean():.4f}  (uniform: 0.0500)")
    print(f"  frac in [0.25, 0.75] = {((u >= 0.25) & (u <= 0.75)).mean():.4f}  (uniform: 0.5000)")