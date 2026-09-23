"""One-off diagnostic for HMM NLL. Run, inspect, delete."""
from __future__ import annotations

import numpy as np

from src.config import load_config
from src.models.hmm_gaussian import HMMGaussianModel
from src.utils.io import load_parquet

RUN = "runs/20260923T152137Z"

cfg = load_config("configs/default.yaml")
X = load_parquet(f"{RUN}/features/X.parquet").to_numpy()
y = load_parquet(f"{RUN}/features/y.parquet").to_numpy()

model = HMMGaussianModel(
    n_states=cfg["models"][-1]["params"]["n_states"],
    covariance_type=cfg["models"][-1]["params"]["covariance_type"],
    n_iter=cfg["models"][-1]["params"]["n_iter"],
    seed=cfg["run"]["seed"],
)
model.fit(X[:96], y[:96])
mu, Sigma = model.predict_distribution(X[96])

np.set_printoptions(precision=5, suppress=True)
print("--- fold 1 (t_test = 96) ---")
print("HMM means_ (regime means):")
print(model.model.means_)
print("HMM covars_ (shape):", model.model.covars_.shape)
print("HMM covars_ (values):")
print(model.model.covars_)
print()
print("posterior at last train point:", model.model.score_samples(y[:96])[1][-1])
print("mixture mu:", mu)
print("y[t_test]:", y[96])
print("diag(Sigma):", np.diag(Sigma))
print("eigs(Sigma):", np.linalg.eigvalsh(Sigma))
print("Mahalanobis (y-mu)' Sigma^-1 (y-mu):",
      float((y[96] - mu) @ np.linalg.solve(Sigma, y[96] - mu)))
print("log|Sigma|:", float(np.linalg.slogdet(Sigma)[1]))
print("NLL:", float(0.5 * (5 * np.log(2 * np.pi)
                          + np.linalg.slogdet(Sigma)[1]
                          + (y[96] - mu) @ np.linalg.solve(Sigma, y[96] - mu))))