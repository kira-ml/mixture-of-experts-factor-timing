# Minimum Viable Baseline — End-to-End Pipeline Architecture

**Project:** moe-factor-timing v2
**Status:** Draft
**Last Updated:** 2026-08-03

---

## 1. Purpose

This document specifies the **smallest end-to-end system** that can produce a valid, reproducible, reportable result under the constraints defined in `01` through `07`.

Its job is to answer a single question before any code is written:

> **What is the minimum set of components, interfaces, and control flow that must exist for the pipeline to be valid end-to-end?**

The MVP is not a scaled-down version of the final system. It is the **backbone** of the final system. Every later addition — models, features, robustness tests — plugs into the interfaces defined here. If the MVP is correct, extensions are additive. If the MVP is wrong, every extension inherits the flaw.

This document is deliberately about **architecture**, not implementation. It defines modules, boundaries, contracts, and data flow. It does not define functions, classes, or file layouts beyond what is needed to make the contracts unambiguous.

---

## 2. Context and Dependencies

- **Depends on:** `00_docs_index.md`, `01_first_principles_problem_decomposition.md`, `02_problem_framing.md`, `03_scope_and_non_goals.md`, `04_assumptions_and_invariants.md`, `05_objective_utility_and_decision_specification.md`, `06_data_information_and_target_contract.md`, `07_evaluation_framework_and_backtesting_protocol.md`
- **Feeds into:** `09_stop_criteria_and_kill_criteria.md`
- **Reference only:** v1 code in `archive/v1/`, tagged `v1.0-final`

Every design decision in this document must be traceable to a decision, invariant, or contract in a prior document. If a design choice cannot be traced, it does not belong in the MVP.

---

## 3. Relationship to v1

v1 is **archived** and treated as a reference implementation, not as a template.

### 3.1 v1 archive location

```
archive/v1/
├── src/                  # v1 source modules
├── main.py               # v1 orchestrator
├── run_moe_torch.py      # v1 PyTorch experiment
├── data/                 # v1 raw + processed data
├── notebooks/            # v1 notebooks
├── scripts/              # v1 utility scripts
├── logs/                 # v1 run logs
├── requirements_v1.txt   # v1 dependencies
├── README_v1.md          # v1 README
├── problem_framing.md    # v1 problem framing
└── TODO.md               # v1 build log
```

### 3.2 v1 isolation rules

- v1 code is **not** on the v2 import path.
- v1 modules are **not** imported by v2 modules.
- v1 data is **not** used by v2 runs.
- v1 is recoverable via the git tag `v1.0-final`.
- v1 is retained for reference and comparison only.

### 3.3 What v2 inherits conceptually from v1

- The domain: US equity factor allocation.
- The overall goal: regime-aware, probabilistic factor timing.
- The baseline idea: compare a probabilistic model against naive baselines.

### 3.4 What v2 explicitly rejects from v1

| v1 choice | v2 replacement |
|-----------|----------------|
| Flat `src/` layout | Ten-stage modular layout |
| Ad hoc allocation rules | Decision rule derived from objective (`05`) |
| Single-shot Sharpe reporting | Layered evaluation with CIs (`07`) |
| Implicit model selection | Explicit separation of selection and evaluation (`07`, Section 10.6) |
| Outputs inside `src/results/` | Immutable `runs/<run_id>/` + curated `results/` |
| Assumed data availability | Vintage-aware contract (`06`) |
| Look-ahead risks in features | Enforced validation checks (`06`, Section 9) |

---

## 4. MVP Philosophy

### 4.1 What "minimum viable" means

Minimum viable means:

1. **End-to-end.** The pipeline runs from raw data to a reportable result without manual intervention.
2. **Valid.** Every invariant in `04` is enforced, every check in `06` runs, and the evaluation protocol in `07` is respected.
3. **Reproducible.** Given the same configuration and the same raw data, the pipeline produces identical results.
4. **Extensible.** Adding a model, a feature, or a robustness dimension does not require restructuring.
5. **Auditable.** Every stage produces artifacts that can be inspected and traced.

Minimum viable does **not** mean "smallest number of lines." It means "smallest set of components that together satisfy the contract."

### 4.2 What the MVP must do

The MVP must, at minimum:

1. Ingest raw data from documented sources with provenance.
2. Enforce the data contract (validation checks V1–V21 in `06`).
3. Construct features and targets that obey the feature contract (`06`, Section 7).
4. Manage walk-forward splits (`06`, Section 10).
5. Fit at least **two models**: one naive baseline (1/N) and one probabilistic model.
6. Produce a predictive distribution for each model at each test time.
7. Solve the constrained decision problem for each model at each test time (`05`, Section 5).
8. Deduct transaction costs (`05`, Section 7).
9. Track portfolio state, wealth, and turnover across time.
10. Evaluate results at Layers L1, L2, L3 (`07`, Sections 5–7).
11. Produce bootstrap confidence intervals for headline metrics (`07`, Section 8.1).
12. Store all outputs with a run manifest (`07`, Section 13).

### 4.3 What the MVP must not do

The MVP must not:

- Include deep learning models.
- Include more than one regime-aware model.
- Include more than one utility function.
- Include more than one cost scenario.
- Include more than three robustness dimensions.
- Include capacity analysis beyond a documented placeholder.
- Include regime stability tests beyond descriptive statistics.
- Include interactive dashboards, APIs, or serving infrastructure.
- Include any code path not reachable from the configuration.
- Import from `archive/v1/`.

These are deferred to post-MVP (Section 13).

---

## 5. Architectural Principles

Eight principles govern the architecture. Every module must respect all eight.

### P1 — Configuration is the only input

A run is fully specified by a configuration object plus a data snapshot. No hidden parameters. No environment-dependent behavior.

### P2 — Raw data is immutable

Raw data is stored once, never overwritten. Every run pulls its own snapshot. Revisions are visible as new snapshots.

### P3 — Contracts are enforced at module boundaries

Every module validates its inputs against the contract of the module that produced them. Failures abort the run with a diagnostic, not a warning.

### P4 — Stages are separable

Each stage (ingest, validate, feature, split, model, decision, backtest, evaluate, report) can be run independently and tested independently. No stage depends on the internal state of another.

### P5 — Determinism by default

Given the same configuration and data snapshot, every stage produces identical output. Random seeds are explicit. No wall-clock behavior.

### P6 — Artifacts are auditable

Every stage writes its outputs to a run-specific directory. Outputs are hashed. Nothing is overwritten within a run.

### P7 — No silent failures

Any error in validation, solver, or model training is either fatal or explicitly recorded. Degraded runs are flagged.

### P8 — Reporting is downstream of evaluation

The reporting stage consumes only stored artifacts. It does not re-run models or recompute metrics. This guarantees that what is reported is what was computed.

---

## 6. System Decomposition

The MVP decomposes into ten stages. Each stage has a contract. Stages communicate only through artifacts.

```
[1]  Ingest   ─►  Raw snapshot + provenance
                      │
                      ▼
[2]  Validate ─►  Validation report + cleaned snapshot
                      │
                      ▼
[3]  Features ─►  Feature matrix + target + manifest
                      │
                      ▼
[4]  Split    ─►  Split manifest (train/test indices)
                      │
                      ▼
[5]  Models   ─►  Predictive distributions per model
                      │
                      ▼
[6]  Decision ─►  Weights + solver diagnostics
                      │
                      ▼
[7]  Backtest ─►  Net returns + turnover + costs
                      │
                      ▼
[8]  Evaluate ─►  Metrics tables + bootstrap results
                      │
                      ▼
[9]  Report   ─►  Figures + summary + disclosures
                      │
                      ▼
[10] Run      ─►  Manifest + validation of end-to-end integrity
```

Stages 1–4 are pre-decision. Stage 5 produces predictions. Stage 6 produces allocations. Stage 7 simulates execution. Stage 8 computes metrics. Stage 9 produces the report. Stage 10 orchestrates and records.

---

## 7. Stage Specifications

### 7.1 Stage 1 — Ingest

**Module:** `src/ingest/`

**Purpose:** Retrieve raw data and store it as an immutable snapshot.

**Inputs:**
- Data source configuration (vendor, endpoint, ticker list, date range)
- Retrieval timestamp

**Outputs:**
- Raw data files per series in `runs/<run_id>/raw/`
- Provenance manifest: source, timestamp, endpoint, series ID, hash

**Contract:**
- Never overwrite existing files. New snapshot per run.
- Record the retrieval timestamp in UTC.
- Record the raw response checksum for each series.
- If a source is unavailable, abort — no fallback to cached data without explicit configuration.

**Invariants enforced:**
- `IV.3` (data provenance is recorded)
- `IV.4` (results reproducible from configuration + raw data)

**Explicitly out of scope for MVP:**
- Automatic retry policies
- Multiple vendors
- Streaming ingestion

---

### 7.2 Stage 2 — Validate

**Module:** `src/validate/`

**Purpose:** Enforce the data contract from `06`, Sections 4–9.

**Inputs:**
- Raw data snapshot from Stage 1
- Validation rules from `06`, Section 9

**Outputs:**
- `validation_report.json` with pass/fail for each check
- Optional: cleaned data snapshot if the run proceeds

**Contract:**
- Every check V1–V21 must run.
- Any check failure aborts the run.
- The validation report is stored with the run and referenced in the manifest.

**Invariants enforced:**
- `I.1`–`I.6` (all information set invariants)
- `IV.3` (data provenance)

**Explicitly out of scope for MVP:**
- Custom user-defined checks
- Anomaly detection
- Data repair beyond documented forward-fill

---

### 7.3 Stage 3 — Features

**Module:** `src/features/`

**Purpose:** Transform validated raw data into model features and targets.

**Inputs:**
- Validated data snapshot from Stage 2
- Feature configuration (lags, windows, transformations, macro series)

**Outputs:**
- Feature matrix `X` with metadata
- Target matrix `y`
- Feature manifest: name, category, source, `reference_date`, `available_at`, transform, dtype, units
- Imputation report: which features used forward-fill, at what rate

**Contract:**
- Every feature obeys rules R1–R7 in `06`, Section 7.2.
- Every feature has an `available_at` timestamp.
- Target construction follows `06`, Section 8.
- No feature may use future information.
- Feature count respects the `T / p ≥ 5` rule (`06`, Section 7.4).

**Invariants enforced:**
- `I.2` (no target leakage)
- `I.5` (no future information in feature engineering)
- `IV.4` (deterministic features)

**Explicitly out of scope for MVP:**
- Feature selection algorithms
- Automatic feature engineering
- Cross-sectional aggregations beyond documented rules

---

### 7.4 Stage 4 — Split

**Module:** `src/split/`

**Purpose:** Generate walk-forward train/test splits.

**Inputs:**
- Feature and target matrices from Stage 3
- Split configuration: window type, minimum training size, test step

**Outputs:**
- Split manifest: for each test time `t`, the training window `[start, t-1]` and test window `[t, t]`
- Optional validation split if nested CV is used

**Contract:**
- Walk-forward only (`06`, Section 10.1).
- Training window ends strictly before test window (invariant `I.6`).
- Split configuration is frozen before evaluation (`07`, Section 10.7).
- Every split is recorded with dates and sizes.

**Invariants enforced:**
- `I.1` (no look-ahead in training data)
- `I.6` (test period strictly after training period)

**Explicitly out of scope for MVP:**
- Combinatorial purged cross-validation
- Rolling windows beyond the default expanding mode

---

### 7.5 Stage 5 — Models

**Module:** `src/models/`

**Purpose:** Fit models on training data and produce predictive distributions on test data.

**Inputs:**
- Training data from Stage 4
- Test features from Stage 4
- Model configuration

**Outputs:**
- Predictive distributions `p̂( r_t | F_t )` for each test time `t`, per model
- Predictive mean `μ_t` and covariance `Σ_t` per model per test time
- For mixture models: gating probabilities `π_t(k)` and expert parameters per `k`
- Training diagnostics: convergence, iteration count, loss trace, runtime
- Model artifacts (fitted parameters, hashes)

**Contract:**
- Every model exposes the same interface: `fit(X_train, y_train)`, `predict_distribution(X_test)`.
- Every model produces a well-defined predictive distribution.
- If a model cannot produce a full covariance, it must document the approximation.
- If a model fails to fit, the run does not silently skip it — it records the failure and either aborts or continues under the documented protocol.

**Models in MVP:**

| Model | Type | Rationale |
|-------|------|-----------|
| 1/N | Deterministic | Trivial baseline. No training required. |
| Persistence | Deterministic | Naive baseline. No training required. |
| Rolling Average | Deterministic | Smoothed baseline. No training required. |
| Momentum | Deterministic | Trend baseline. No training required. |
| Ridge Regression | Probabilistic | Produces `μ` and `Σ` under a documented residual model. |
| One regime-aware model | Probabilistic | Either HMM-Gaussian or SimpleMoE (linear experts). |

**Explicitly out of scope for MVP:**
- PyTorch / deep learning models
- Ensemble methods
- Bayesian hierarchical models
- Multi-output deep models

**Invariants enforced:**
- `IV.2` (random seeds fixed)
- `IV.4` (deterministic fitting)
- `C7` (backtest is not used for model selection)

---

### 7.6 Stage 6 — Decision

**Module:** `src/decision/`

**Purpose:** Map each model's predictive distribution into portfolio weights.

**Inputs:**
- Predictive distribution from Stage 5 (`μ_t`, `Σ_t`, or full mixture)
- Previous weights `w_{t-1}`
- Constraint configuration
- Cost configuration
- Utility configuration (mean-variance, default)
- Risk aversion `λ`

**Outputs:**
- Chosen weights `w_t*` for each model at each test time
- Solver diagnostics: status, iteration count, objective value
- Constraint check report: all constraints satisfied (yes/no)
- Fallback events (if any)

**Contract:**
- Decision rule is exactly the one specified in `05`, Section 5.
- No ad hoc rules are permitted.
- All constraints in `05`, Section 6.1 are enforced.
- If the solver fails, the fallback rule in `05`, Section 9.3 applies.
- Fallback frequency is capped at 5%.

**Invariants enforced:**
- `II.1` (decision rule derived from objective)
- `II.2` (constraints enforced)
- `II.5` (risk penalty active)

**Explicitly out of scope for MVP:**
- Multi-period optimization
- Robust optimization
- Distributionally robust allocation

---

### 7.7 Stage 7 — Backtest

**Module:** `src/backtest/`

**Purpose:** Simulate the walk-forward portfolio evolution with costs.

**Inputs:**
- Weights `w_t*` from Stage 6
- Realized returns `r_t`
- Cost configuration

**Outputs:**
- Net portfolio return per period `R_t = w_t*^T r_t − C_t`
- Cumulative wealth `W_t`
- Turnover per period
- Cost per period
- Fallback events
- Portfolio state trace

**Contract:**
- Costs are deducted inside the backtest, not post hoc (`07`, Section 10.3).
- Cold start uses cash or 1/N, declared a priori.
- Fallback events are recorded.
- Every output has a stable ordering (by date).

**Invariants enforced:**
- `II.3` (costs deducted from realized returns)
- `II.4` (turnover reported)
- `I.4` (portfolio state lags correctly)

**Explicitly out of scope for MVP:**
- Capacity-constrained backtest
- Regime-conditional costs
- Intra-period rebalancing

---

### 7.8 Stage 8 — Evaluate

**Module:** `src/evaluate/`

**Purpose:** Compute the evaluation metrics defined in `07`.

**Inputs:**
- Backtest outputs from Stage 7
- Predictive distributions from Stage 5
- Realized returns
- Benchmark outputs

**Outputs:**
- **Layer 1:** NLL, CRPS, PIT histogram, calibration plot, interval score
- **Layer 2:** RMSE, MAE, R² per asset and aggregated
- **Layer 3:** Expected utility, annualized return, volatility, Sharpe, Sortino, MaxDD, Calmar, win rate, skewness, kurtosis, turnover, cost drag, capacity estimate
- **Layer 4:** Bootstrap confidence intervals for all Layer 3 metrics
- **Layer 5:** Robustness outputs across declared dimensions

**Contract:**
- Primary predictive metric is NLL (`07`, Section 5.6).
- Primary decision metric is expected utility (`07`, Section 7.5).
- Every headline metric has a bootstrap confidence interval (`07`, Section 8.1).
- Benchmark results use the same cost model, constraints, and calendar.
- If `N_trials > 1`, deflated Sharpe is computed.

**Invariants enforced:**
- `III.1` (backtest not used for model selection)
- `III.4` (confidence statements for every metric)
- `III.5` (multiple-testing correction)
- `III.6` (robustness checks reported)

**Explicitly out of scope for MVP:**
- Full DM / SPA test suite (only DSR required in MVP)
- Cross-sectional factor regressions
- Attribution analysis

---

### 7.9 Stage 9 — Report

**Module:** `src/report/`

**Purpose:** Produce the final reportable artifacts.

**Inputs:**
- All stored artifacts from Stages 1–8

**Outputs:**
- Summary table: metrics for each model and benchmark, with confidence intervals
- Figures: cumulative returns, drawdown, turnover, calibration, robustness distributions
- Run manifest (`manifest.json`) with all hashes and configuration
- Reproducibility statement
- Disclosure statement (non-goals from `03`, Section 9)

**Contract:**
- Report consumes only stored artifacts. No recomputation.
- Every reported number is traceable to an artifact.
- Every claim follows the reporting standard in `07`, Section 12.
- Missing inputs cause the report to fail, not silently omit items.

**Invariants enforced:**
- `IV.1` (environment captured)
- `IV.4` (results reproducible)
- `IV.5` (configurations versioned)
- `IV.6` (no silent code changes)

**Explicitly out of scope for MVP:**
- Interactive dashboards
- PDF generation beyond a simple markdown/CSV report
- Multi-run comparison reports

---

### 7.10 Stage 10 — Run (Orchestrator)

**Module:** `src/run/`

**Purpose:** Orchestrate Stages 1–9 and produce the run manifest.

**Inputs:**
- Configuration file (`configs/*.yaml`)
- Command-line arguments (only to select the configuration and override a small allow-list of fields)

**Outputs:**
- `runs/<run_id>/manifest.json`
- End-to-end integrity check report

**Contract:**
- Stages are invoked in order.
- If any stage aborts, the run aborts.
- The manifest is written only after all stages complete.
- Any stage's outputs are stored under `runs/<run_id>/`.
- Run IDs are timestamp-based and unique.
- A run never overwrites a previous run's directory.

**Invariants enforced:**
- `P1` (configuration is the only input)
- `P6` (artifacts auditable)
- `P7` (no silent failures)
- `IV.1`, `IV.5`, `IV.6`

---

## 8. Data Flow

The pipeline is a strict left-to-right flow. No stage reads from a later stage's outputs.

```
Raw Data (immutable)
   │
   ▼
Validated Data
   │
   ▼
Features + Targets
   │
   ▼
Splits (train/test indices)
   │
   ├─────► Baseline Models ──► Predictive Distributions
   │
   └─────► Probabilistic Model ──► Predictive Distributions
                                        │
                                        ▼
                                   Decision Rule
                                        │
                                        ▼
                                   Weights w_t*
                                        │
                                        ▼
                                   Backtest
                                        │
                                        ▼
                                   Metrics
                                        │
                                        ▼
                                   Report
```

### 8.1 Artifact boundaries

Every arrow represents a serialized artifact:

| From | To | Artifact |
|------|----|----------|
| Ingest | Validate | Raw snapshot + provenance manifest |
| Validate | Features | Validation report + cleaned snapshot |
| Features | Split | Feature matrix + target + feature manifest |
| Split | Models | Split manifest (dates) |
| Models | Decision | Predictive distribution files |
| Decision | Backtest | Weights + solver diagnostics |
| Backtest | Evaluate | Net returns + turnover + costs |
| Evaluate | Report | Metrics tables + bootstrap results |
| Report | Run | Final report + figures |
| Run | — | Manifest |

Every artifact is written to `runs/<run_id>/`. Artifacts are never overwritten within a run.

---

## 9. Control Flow

### 9.1 Single run

A run proceeds through Stages 1–10 in order.

### 9.2 Failure handling

| Failure | Response |
|---------|----------|
| Ingest fails | Abort run. Record reason. |
| Validation check fails | Abort run. Record which check. |
| Feature construction fails | Abort run. Record which feature. |
| Model fit fails | Record failure. Abort if it is the only probabilistic model. |
| Solver fails | Apply fallback (`05`, 9.3). Record. Continue. |
| Fallback frequency > 5% | Abort run. |
| Evaluation fails | Abort run. Record which metric. |
| Report fails | Abort run. No partial report. |

### 9.3 No resume

A run is atomic. Partial runs are not resumed. Restarting requires a new run ID.

### 9.4 Parallelism

MVP runs sequentially. Parallelization is deferred.

### 9.5 Determinism

Every stage is deterministic. Random seeds are fixed per stage and recorded. If a stage is non-deterministic (e.g., bootstrap), the seed is recorded and reproducibility is checked by re-running.

---

## 10. Configuration and Manifest

### 10.1 Configuration object

The configuration defines everything needed for a run. It is a single artifact, hashed before the run begins.

**Location:** `configs/<name>.yaml`

**Minimum fields:**

```yaml
run:
  run_id: null  # auto-generated
  timestamp: null  # auto-generated

data:
  sources:
    yfinance:
      tickers: [SPY, IWD, MTUM, QUAL, USMV]
      frequency: monthly
    fred:
      series: [CPIAUCSL, INDPRO, UNRATE, T10Y2Y, GS10, GS2, VIXCLS]
      vintage: alfed
  universe: [SPY, IWD, MTUM, QUAL, USMV]
  cash_proxy: BIL
  start_date: "2013-08-01"
  end_date: null  # auto-set to last available month

features:
  lagged_returns_lags: 12
  rolling_windows: [3, 6, 12]
  macro_series: [CPIAUCSL, INDPRO, UNRATE, T10Y2Y, GS10, GS2, VIXCLS]
  transformations: [diff, log_diff, zscore]
  imputation:
    method: ffill_only
    max_rate: 0.05

split:
  type: expanding
  min_train: 96
  test_step: 1

models:
  - name: naive_1n
    type: deterministic
  - name: persistence
    type: deterministic
  - name: rolling_avg
    type: deterministic
    params: {window: 12}
  - name: momentum
    type: deterministic
    params: {decay: 0.9, window: 12}
  - name: ridge
    type: probabilistic
    params: {alpha: 1.0}
  - name: moe_simple
    type: regime_aware
    params: {n_experts: 4, n_iterations: 100}

decision:
  utility: mean_variance
  risk_aversion: 10
  constraints:
    long_only: true
    max_weight: 0.40
    turnover_cap: 0.50
    min_deployment: 0.0
  cost:
    mode: assumed
    bps: 10.0
  solver:
    name: cvxpy
    tolerance: 1.0e-6
    fallback_max_rate: 0.05

evaluation:
  primary_predictive_metric: nll
  primary_decision_metric: expected_utility
  bootstrap:
    method: stationary_block
    block_length: 3
    n_repetitions: 1000
    seed: 42
  deflated_sharpe:
    enabled: true

robustness:
  dimensions:
    - min_train: [60, 96, 132]
    - n_experts: [2, 4, 5]
    - cost_bps: [5, 10, 20]

random_seeds:
  global: 42
  bootstrap: 42
  models: 42

reporting:
  output_dir: results/<run_id>/
  include_manifest: true
  include_disclosures: true
```

### 10.2 Manifest

The manifest is written once, at the end of a run. It contains:

- `run_id`
- Configuration hash
- Data snapshot hash
- Code commit hash
- Python version, OS
- Package versions (key libraries)
- Random seeds
- Start and end timestamps
- Output paths and hashes per artifact
- Validation report reference
- End-to-end integrity check result

The manifest is the single source of truth for what was run.

### 10.3 Configuration validation

Before any stage runs, the configuration is validated against the schema defined in this section. Unknown fields cause an abort. Missing required fields cause an abort.

---

## 11. Directory Structure (Repository Contract)

The repository layout is a **contract**, not a suggestion. It was established in the v1→v2 migration and must be respected.

### 11.1 Top level

```
mixture-of-experts-factor-timing/
├── archive/
│   └── v1/                        # v1 code, data, docs (tagged v1.0-final)
├── configs/                       # Run configurations (YAML)
│   └── default.yaml
├── docs/                          # Pre-code documentation (00–09)
├── results/                       # Curated, committed outputs
│   └── .gitkeep
├── runs/                          # Immutable run outputs (git-ignored)
│   └── <run_id>/
├── src/                           # v2 pipeline modules
│   ├── __init__.py
│   ├── ingest/
│   ├── validate/
│   ├── features/
│   ├── split/
│   ├── models/
│   ├── decision/
│   ├── backtest/
│   ├── evaluate/
│   ├── report/
│   └── run/
├── tests/                         # Contract and falsification tests
├── .gitignore
├── README.md                      # v2 project overview
└── venv/                          # Local virtualenv (git-ignored)
```

### 11.2 `runs/<run_id>/` internal structure

```
runs/<run_id>/
├── raw/                           # Immutable raw snapshot
├── validated/                     # Cleaned snapshot
├── features/                      # X, y, feature manifest
├── splits/                        # Split manifest
├── predictions/                   # Predictive distributions
├── weights/                       # Decision outputs
├── backtest/                      # Portfolio trace
├── metrics/                       # Evaluation outputs
├── report/                        # Report artifacts + figures
├── validation_report.json         # Stage 2 output
└── manifest.json                  # Written at end of Stage 10
```

### 11.3 Directory rules

- `runs/` is **never** committed to git.
- `runs/` is **never** modified after a run completes.
- `results/` is curated and committed.
- `archive/` is never imported by `src/`.
- `configs/` files are the only source of run parameters.
- `src/` contains only the ten modules listed in Section 7.
- `tests/` contains only contract and falsification tests.

### 11.4 `.gitignore` contract

The following must be ignored:

```
# v2 run outputs (immutable, never committed)
runs/

# Virtual environments
venv/
.venv/
env/

# Python
__pycache__/
*.py[cod]
*.egg-info/

# IDE and OS
.vscode/
.idea/
.DS_Store
Thumbs.db

# Secrets
.env
.env.local

# Archived v1 data (recoverable via v1.0-final tag)
archive/v1/data/
```

`results/` is **not** ignored. It is committed but curated.

---

## 12. Module Boundaries

The MVP defines the following module boundaries. Implementation details are deferred, but the boundaries are frozen.

| Module | Path | Responsibility | Input | Output |
|--------|------|---------------|-------|--------|
| `ingest` | `src/ingest/` | Fetch raw data | Config | Raw snapshot + provenance |
| `validate` | `src/validate/` | Enforce contract | Raw snapshot + rules | Validation report |
| `features` | `src/features/` | Build features + targets | Validated data | Feature matrix + manifest |
| `split` | `src/split/` | Generate walk-forward splits | Features + split config | Split manifest |
| `models` | `src/models/` | Fit models, produce distributions | Train + test data | Predictive distributions |
| `decision` | `src/decision/` | Solve allocation | Distributions + state + config | Weights + diagnostics |
| `backtest` | `src/backtest/` | Simulate portfolio | Weights + returns + costs | Net returns + trace |
| `evaluate` | `src/evaluate/` | Compute metrics | Backtest + distributions | Metrics tables |
| `report` | `src/report/` | Produce reportable artifacts | All artifacts | Report + figures |
| `run` | `src/run/` | Orchestrate | Config | Run manifest |

### 12.1 Interface rules

- Each module exposes exactly one public interface (a function or a small set of functions).
- Modules do not call each other directly; `src/run/` invokes them in order.
- No module imports from another module except the standard library or shared utilities (deferred).
- No module reads from `runs/` except the one it owns; the `run` module mediates cross-stage access.

---

## 13. Deferred to Post-MVP

The following are explicitly **out of scope** for the MVP. They are noted so they are not forgotten, and so the MVP's interface is designed to accept them.

### 13.1 Models

- PyTorch MoE, LSTM gating, deep ensembles.
- Bayesian hierarchical models.
- Gradient boosting, XGBoost.
- Ensemble or stacking methods.

### 13.2 Features

- Cross-sectional aggregations.
- Text-based sentiment.
- Alternative data (satellite, credit card, etc.).
- Regime-conditional feature transformations.

### 13.3 Decision

- CVaR-based allocation.
- Multi-period optimization.
- Robust / distributionally robust allocation.
- Leverage, shorting.
- Regime-conditional constraints.

### 13.4 Evaluation

- Full DM / SPA test suite.
- Capacity curves.
- Attribution analysis.
- Cross-sectional regressions.

### 13.5 Infrastructure

- Parallel runs.
- Distributed training.
- Docker.
- Experiment tracking (MLflow, W&B).
- Cloud deployment.

### 13.6 Reporting

- Interactive dashboards.
- Automated PDF generation.
- Multi-run comparison reports.

---

## 14. Explicitly Rejected Designs

The following designs were considered and rejected. They are documented so they are not reintroduced silently.

| Design | Reason for Rejection |
|--------|---------------------|
| A single monolithic `main.py` | Violates P4 (separability). Made v1 undiagnosable. |
| Training on the same data used for evaluation | Violates `III.1` and `I.6`. |
| Ad hoc allocation rules (e.g., magnitude-weighted) | Violates `II.1`. |
| Reporting Sharpe without confidence intervals | Violates `III.4`. |
| Selecting models on the evaluation window | Violates `III.1`. |
| Silent fallbacks in solvers or feature construction | Violates P7. |
| Global (full-sample) z-scores or scalers | Violates `I.5`. |
| Storing results without a manifest | Violates `IV.3`, `IV.4`. |
| Using data revisions without vintages | Violates `I.3`. |
| Running robustness on the same seeds as evaluation | Violates `IV.2`. |
| Committing v1 code into v2 `src/` | Mixing histories. v1 goes to `archive/v1/`. |
| Keeping v1 files at repository root | Ambiguity about which version is active. |
| Writing outputs into `src/` | Violates P6 and the run-artifact contract. |

---

## 15. Ties to Other Documents

| Concern | Where specified |
|---------|-----------------|
| Objective and utility | `05` |
| Constraints and costs | `05`, Sections 6–7 |
| Data sources and vintages | `06` |
| Feature contract | `06`, Section 7 |
| Target contract | `06`, Section 8 |
| Validation checks | `06`, Section 9 |
| Evaluation metrics | `07`, Sections 5–9 |
| Backtesting protocol | `07`, Section 10 |
| Reporting standard | `07`, Section 12 |
| Stop and kill criteria | `09` |

Conflicts are resolved by updating the affected document, not by silent override.

---

## 16. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Ten stages: ingest, validate, features, split, models, decision, backtest, evaluate, report, run. | Matches the four sub-problems from `01` with data, evaluation, and orchestration layers. |
| D2 | Configuration is the only input to a run. | Reproducibility and auditability. |
| D3 | Raw data is immutable. | Prevents silent drift. |
| D4 | Contracts are enforced at module boundaries. | Prevents silent failures. |
| D5 | MVP includes exactly one regime-aware model. | Sufficient to validate the pipeline without diluting effort. |
| D6 | MVP includes only mean-variance utility. | Additional utilities deferred. |
| D7 | MVP includes only 10 bps assumed cost. | Calibration deferred. |
| D8 | MVP includes only expanding-window splits. | Rolling windows deferred. |
| D9 | MVP requires the full validation check suite (V1–V21). | No partial validation. |
| D10 | MVP requires bootstrap CIs and DSR. | Honest inference is not optional. |
| D11 | Report consumes only stored artifacts. | Prevents post hoc reanalysis. |
| D12 | Run directory is atomic and never modified. | Auditability. |
| D13 | Falsification tests are required. | Contract tests alone are not sufficient. |
| D14 | Directory structure is a contract. | Prevents ad hoc file placement. |
| D15 | Rejected designs are documented. | Prevents silent reintroduction. |
| D16 | v1 lives in `archive/v1/` and is tagged `v1.0-final`. | Preserves v1 for reference without contaminating v2. |
| D17 | v1 data is excluded from version control. | Recoverable via tag; keeps repository small. |
| D18 | `runs/` is git-ignored; `results/` is curated and committed. | Separates ephemeral outputs from published artifacts. |

---

## 17. Testing Strategy

The MVP is validated by contract-level tests, not implementation-level tests.

### 17.1 Contract tests

| Test | Validates |
|------|-----------|
| Determinism | Same config → identical artifact hashes |
| Provenance | Every raw series has a recorded source and timestamp |
| Information set | No feature has `available_at > rebalance_date` |
| Target leakage | No feature correlates with target above ceiling |
| Split integrity | Every train window ends before its test window |
| Constraint enforcement | No output weight violates the constraint set |
| Solver fallback | Fallback events are recorded and capped |
| Cost deduction | Costs are subtracted inside the backtest |
| Metric reproducibility | Bootstrap CIs reproduce under fixed seed |
| Manifest completeness | All required fields present and consistent |

### 17.2 Falsification tests

Each stage has at least one test that would fail if the contract were violated:

- Inject a feature with a future timestamp → validation should abort.
- Inject a target into a feature → leakage check should abort.
- Force a solver failure → fallback should trigger and be recorded.
- Shuffle the training data → determinism test should fail.
- Remove the manifest → report should fail.

### 17.3 Integration test

The MVP passes an integration test if:

1. A full run completes on the default configuration.
2. All artifacts are produced.
3. The manifest is complete.
4. A second run with the same configuration produces identical hashes.

---

## 18. Open Questions

1. Should Stage 2 (Validate) write a "cleaned" snapshot, or should cleaning be part of Stage 3 (Features)?
2. Should the split manifest be generated once and reused across models, or regenerated per model?
3. Should the MVP include a null model (e.g., random weights) as a decision-quality sanity check?
4. Is the `runs/` directory model sufficient for multi-run comparison, or does a separate comparison artifact need to exist in the MVP?
5. Should the MVP support running a subset of models (e.g., `--models 1n ridge`) without violating the manifest contract?
6. Should the MVP have a "dry-run" mode that validates configuration and data without running models?
7. Should the report stage produce a Markdown report, a CSV set, or both?
8. Is capacity analysis in scope for the MVP as a placeholder, or should it be entirely deferred?
9. Is the `T / p ≥ 5` rule enforced at Stage 3 or Stage 5?
10. Should the MVP record intermediate artifacts (e.g., model parameters) as separate files, or bundle them into the manifest?
11. Should the MVP include a minimal pre-commit hook that checks for invariant violations before any commit?
12. Does the MVP need a "compare runs" script, or is manual inspection of manifests sufficient?
13. Should the "one regime-aware model" be HMM-Gaussian or SimpleMoE? The choice affects the interface for mixture distributions.
14. Should the MVP include a minimal PyTorch MoE behind a flag for parity with v1, or exclude it entirely?
15. Is the 5% fallback threshold tested in the MVP, or only enforced in production runs?
16. Should `src/utils.py` exist as a shared utility module, or should utilities live inside each stage module?
17. Should `pyproject.toml` be added for packaging, or is a plain `requirements.txt` sufficient for v2?
18. Should the `run` module expose a CLI, a Python API, or both?

---

## 19. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial MVP end-to-end pipeline architecture for v2. |
| 0.2 | 2026-08-03 | Updated after v1→v2 migration. Added v1 archive contract, actual repository layout, `.gitignore` contract, ten-stage flow with explicit `run` module, and post-migration open questions. |

---

## 20. References

- `00_docs_index.md` — documentation map and conventions.
- `01_first_principles_problem_decomposition.md` — primitive decomposition.
- `02_problem_framing.md` — formal problem statement.
- `03_scope_and_non_goals.md` — scope boundaries.
- `04_assumptions_and_invariants.md` — assumptions and invariants.
- `05_objective_utility_and_decision_specification.md` — objective, utility, decision rule.
- `06_data_information_and_target_contract.md` — data contract.
- `07_evaluation_framework_and_backtesting_protocol.md` — evaluation protocol.
- `archive/v1/` — v1 implementation, tagged `v1.0-final`.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5), 94–107.
