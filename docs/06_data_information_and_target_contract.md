# Data, Information, and Target Contract

**Project:** moe-factor-timing v2
**Status:** Draft
**Last Updated:** 2026-08-03

---

## 1. Purpose

This document defines the **data contract** for v2.

It specifies:

1. What data is used.
2. Where it comes from.
3. When it is considered available.
4. How it enters the model.
5. What the prediction target is.
6. What is forbidden.

Its job is to eliminate look-ahead bias, target leakage, and silent data drift — three failure modes that invalidate backtests without producing any visible error.

The contract is a **binding interface** between the data pipeline and every downstream component. If a model, feature, or evaluation step uses data not permitted by this document, the run is invalid.

The rule is simple:

> **At time `t`, only information that was actually observable at time `t` may enter the decision.**

Every clause in this document exists to enforce that rule.

---

## 2. Context and Dependencies

- **Depends on:** `00_docs_index.md`, `01_first_principles_problem_decomposition.md`, `02_problem_framing.md`, `03_scope_and_non_goals.md`, `04_assumptions_and_invariants.md`, `05_objective_utility_and_decision_specification.md`
- **Feeds into:** `07_evaluation_framework_and_backtesting_protocol.md`, `08_minimum_viable_baseline_end_to_end_pipeline_architecture.md`
- **Reference only:** v1 `problem_framing.md`, v1 `README.md`

This document assumes the decision problem and utility are already specified in `05`. The data contract exists to serve that decision problem, not the other way around.

---

## 3. Contract Overview

The contract has five parts:

1. **Universe** — what can be traded.
2. **Information set** — what can be observed at time `t`.
3. **Features** — how observations become model inputs.
4. **Target** — what is predicted.
5. **Validation** — how the pipeline ensures the contract is respected.

Each part is a hard boundary. Violations invalidate the run (invariant `I.1`–`I.6` in `04_...`).

---

## 4. Universe Contract

### 4.1 Tradable assets

The investable universe is fixed and declared before any evaluation.

| ID | Ticker | Asset class | Role |
|----|--------|-------------|------|
| `SPY` | SPY | US large-cap equity | Market |
| `IWD` | IWD | US large-cap value | Value |
| `MTUM` | MTUM | US large-cap momentum | Momentum |
| `QUAL` | QUAL | US large-cap quality | Quality |
| `USMV` | USMV | US minimum volatility | Low Volatility |
| `CASH` | BIL (or equivalent) | 1–3 month T-bills | Risk-free / cash |

### 4.2 Universe invariants

- The universe is **fixed** before evaluation. No asset is added or removed during a run.
- VIX is **not** in the universe. VIX spot is not tradable (`03`, Section 4.1). VIX may appear as a **feature** only.
- Every asset must have a continuous price history over the training and evaluation window. If not, either the start date moves forward or the asset is excluded a priori, not during the run.
- ETF tickers must be the ones actually traded on the exchange. Mutual-fund proxies are not acceptable.

### 4.3 Cash treatment

`CASH` represents the risk-free asset. Its return is the monthly T-bill return from FRED (`TB3MS` or `DTB3`, resampled to monthly).

Two options exist for cash treatment (see `05`, Open Question 1):

- **Option A (default):** Cash is a residual. `Σ w_i ≤ 1` with `w_cash = 1 − Σ w_i`. Return on cash is the risk-free rate.
- **Option B:** Cash is an explicit asset in the optimizer.

The choice must be declared before evaluation and applied consistently.

---

## 5. Information Set Contract

### 5.1 Definition of `F_t`

At the start of rebalancing period `t`, the information set `F_t` contains **only**:

| Item | Source | Availability rule |
|------|--------|-------------------|
| Monthly returns for all assets, `t-L` through `t-1` | yfinance / vendor | Last trading day of `t-1` |
| Realized volatility estimates | Derived from returns | Same |
| Realized correlation estimates | Derived from returns | Same |
| Macroeconomic indicators (as published) | FRED / ALFRED | Vintage available at time `t-1` |
| Current portfolio weights `w_{t-1}` | Internal state | End of `t-1` |
| Current cash balance | Internal state | End of `t-1` |

### 5.2 What is excluded from `F_t`

- Any same-period (`t`) return, price, or volume.
- Any revised macro data that was not published at `t-1`.
- Any feature that uses future data in its construction.
- Any cross-sectional aggregate that includes same-period observations.
- Any forward-filled value that used post-`t` data.

### 5.3 Availability rule

A series is "available at `t-1`" if and only if its first official publication timestamp is at or before the last business day of period `t-1`.

For monthly data released mid-month (e.g., CPI for month `m` released mid-month `m+1`), the reference date is the **reference month**, not the release month. The vintage is the revision available at the time of the decision.

### 5.4 Information invariants

- **I.1** — No look-ahead in training data.
- **I.2** — No target leakage.
- **I.3** — Vintage-aware macro features.
- **I.4** — Lagged portfolio state only.
- **I.5** — No future information in feature engineering.
- **I.6** — Test period strictly after training period.

All six must be machine-checkable. See Section 9.

---

## 6. Data Sources Contract

### 6.1 Asset returns

| Field | Value |
|-------|-------|
| **Source** | yfinance (baseline) or a documented vendor |
| **Frequency** | Daily prices, resampled to monthly |
| **Resampling rule** | Last trading day of each calendar month |
| **Adjustment** | Total return (adjusted close) |
| **Time zone** | US/Eastern |
| **Retrieval timestamp** | Recorded with each run |

**Rule:** Resampling uses the **last trading day** of the month, not the last calendar day. If the last trading day is a holiday, the prior trading day is used.

**Rationale:** Monthly returns must be computable from prices a live investor could observe.

### 6.2 Macroeconomic data

| Field | Value |
|-------|-------|
| **Source** | FRED and ALFRED (vintages) |
| **Frequency** | As published (monthly, quarterly) |
| **Vintage policy** | ALFRED vintage as of the rebalancing date |
| **Fallback** | If ALFRED vintage unavailable, series is lagged by a documented, conservative amount |
| **Retrieval timestamp** | Recorded with each run |

**Candidate series (initial set, non-final):**

| Series | Description | Frequency | Lag rule |
|--------|-------------|-----------|----------|
| `CPIAUCSL` | CPI, all urban consumers | Monthly | 1-month publication lag + vintage |
| `INDPRO` | Industrial production index | Monthly | 1-month lag + vintage |
| `UNRATE` | Unemployment rate | Monthly | 1-month lag + vintage |
| `T10Y2Y` | 10Y–2Y Treasury spread | Daily → monthly | Last available in `t-1` |
| `GS10` | 10Y Treasury yield | Monthly | Last available in `t-1` |
| `GS2` | 2Y Treasury yield | Monthly | Last available in `t-1` |
| `VIXCLS` | VIX close | Daily → monthly | Last available in `t-1` |

The final set of macro features is documented in Section 7 and may shrink during the MVP build.

### 6.3 Risk-free rate

| Field | Value |
|-------|-------|
| **Source** | FRED (`DTB3` or `TB3MS`) |
| **Frequency** | Daily → monthly |
| **Rule** | Monthly average or last available value, documented |

### 6.4 Data provenance rules

- Every raw series is stored with:
  - Source name
  - Retrieval timestamp (UTC)
  - Endpoint or file path
  - Series identifier
  - Transformation applied (if any)
- Raw data is never overwritten. Each run pulls its own snapshot.
- If a vendor revises historical data, the revision is visible in the provenance record.
- If provenance is missing for any series, the run is invalid.

---

## 7. Feature Contract

### 7.1 Feature categories

Features fall into three categories. Each category has its own rules.

**Category 1 — Lagged returns.**
- Monthly returns for lags `1..L` of each asset.
- Default `L = 12`.
- Lagged values only, never contemporaneous.

**Category 2 — Derived statistics.**
- Rolling volatility (window `W ∈ {3, 6, 12}` months, lagged).
- Rolling correlation (window `W ∈ {12}` months, lagged).
- Momentum signals (EWMA-based, lagged).
- Z-scores of returns (rolling, lagged).

**Category 3 — Macro features.**
- Level and lagged values of the macro series in Section 6.2.
- Transformations (diffs, log-diffs, z-scores) use only past data.
- Vintages as of `t-1` only.

### 7.2 Feature construction rules

Every feature must satisfy:

- **R1 — Lag.** Any feature derived from a series at time `s` must use data with reference date `≤ s-1`.
- **R2 — No target leakage.** No feature may be constructed using any part of the target `r_t`.
- **R3 — No cross-sectional leakage.** Any feature that aggregates across assets at time `s` uses only asset data available at `s-1`.
- **R4 — Rolling stats.** Rolling windows use only observations that would have been available at the feature timestamp.
- **R5 — Vintage.** Macro features use the vintage available at `t-1`, not the latest revision.
- **R6 — Determinism.** Given the same raw inputs and the same configuration, features are identical.
- **R7 — No forward fill from the future.** Forward-fill is allowed only with `ffill()` semantics (using past data), never `bfill()`.

### 7.3 Feature schema

Each feature is stored with metadata:

| Field | Description |
|-------|-------------|
| `name` | Unique feature identifier |
| `category` | `lagged_return`, `derived`, `macro` |
| `source` | Source series or computation |
| `reference_date` | Reference date of the underlying data |
| `available_at` | Earliest date the value was observable |
| `transform` | Transformation applied |
| `dtype` | Data type |
| `units` | Units or normalized scale |

### 7.4 Feature count budget

The MVP feature set is documented in `08_...`. The total number of features must remain consistent with the sample size and model complexity:

- With `T` training months and `p` features, require `T / p ≥ 5` as a rule of thumb.
- If `p` exceeds this bound, features are removed by a documented, a priori rule (not by looking at evaluation performance).

### 7.5 Prohibited features

The following are explicitly forbidden:

- Contemporaneous returns of any asset.
- Any feature using the current period's macro release if it was published mid-period.
- Any rolling statistic that includes the current period.
- Any feature computed from the full sample (e.g., global z-scores).
- Any feature derived from the target or its future values.
- Any feature whose construction depends on the evaluation outcome.
- VIX as a target or allocation asset.

---

## 8. Target Contract

### 8.1 Target definition

The prediction target is the vector of **next-month total returns** for the assets in the investable universe:

```
y_t = [ r_{t, SPY}, r_{t, IWD}, r_{t, MTUM}, r_{t, QUAL}, r_{t, USMV} ]
```

where `r_{t, i}` is the monthly return of asset `i` over period `t`.

`CASH` is not a prediction target. Its return is deterministic given the risk-free rate.

### 8.2 Target horizon

- **Horizon:** 1 month.
- **Rationale:** Matches rebalancing frequency (`05`, Section 6).
- **Deferred:** Multi-horizon targets.

### 8.3 Target construction

- Return is computed from adjusted close prices.
- Price for month `t` is the last trading day of month `t`.
- Price for month `t-1` is the last trading day of month `t-1`.
- `r_t = (P_t / P_{t-1}) − 1` in total-return terms.

### 8.4 Target invariants

- Target is computed **after** the decision is made. It is not observable at `t-1`.
- No feature may include any component of the target.
- No imputation of the target is permitted. Missing target values drop the sample.

---

## 9. Validation Contract

The contract is enforced by validation checks. Each check is a binary pass/fail. Any failure invalidates the run.

### 9.1 Data validation checks

| ID | Check | Failure action |
|----|-------|----------------|
| V1 | All asset series are present over the full window | Abort run |
| V2 | No NaN in features or targets after cleaning | Abort run |
| V3 | All dates are monotonic and unique | Abort run |
| V4 | All features are numeric (no object dtype) | Abort run |
| V5 | All features have finite values | Abort run |
| V6 | Macro features match ALFRED vintage for their reference month | Abort run |
| V7 | Asset returns reconcile with raw prices | Abort run |

### 9.2 Information set checks

| ID | Check | Failure action |
|----|-------|----------------|
| V8 | Every feature has `available_at ≤ rebalance_date` | Abort run |
| V9 | No feature uses the target period | Abort run |
| V10 | No feature uses a global rolling statistic | Abort run |
| V11 | No feature constructed from `bfill()` | Abort run |
| V12 | Portfolio state at `t` reflects decisions from `t-1` | Abort run |

### 9.3 Target leakage checks

| ID | Check | Failure action |
|----|-------|----------------|
| V13 | Correlation between each feature and target does not exceed a documented ceiling (default 0.95) | Warn; investigate |
| V14 | Target does not appear in any feature name or source | Abort run |
| V15 | No feature is computable only after the target is known | Abort run |

### 9.4 Split checks

| ID | Check | Failure action |
|----|-------|----------------|
| V16 | Training window ends strictly before test window | Abort run |
| V17 | No sample from the test period appears in training | Abort run |
| V18 | Validation window (if used) is strictly between training and test | Abort run |

### 9.5 Provenance checks

| ID | Check | Failure action |
|----|-------|----------------|
| V19 | All raw series have recorded provenance | Abort run |
| V20 | Provenance timestamps precede the run start | Abort run |
| V21 | Configuration hash matches the one recorded in the run | Abort run |

### 9.6 Validation reporting

Every run emits a validation report containing:

- All check IDs
- Pass/fail status
- Number of rows, features, and assets
- Timestamp of each data pull
- A hash of the input data
- A hash of the configuration

The report is stored alongside results. Without it, results are not reportable.

---

## 10. Split Protocol

### 10.1 Walk-forward evaluation

The only permissible evaluation is walk-forward (`04`, invariant `I.6`):

- **Training window type:** Expanding (default) or rolling (variant).
- **Minimum training size:** Documented; default 96 months (`02`, rationale).
- **Validation window:** Optional; if used, strictly between training and test.
- **Test step:** 1 month.
- **Rebalancing:** Monthly.

### 10.2 Split invariants

- No shuffling.
- No random sampling.
- No data leak across split boundaries.
- Every split has a recorded start and end date.

### 10.3 Model selection split

Model selection (choice of `K`, hyperparameters, feature set) uses one of:

- **Option A:** Nested walk-forward cross-validation inside the training window.
- **Option B:** A held-out validation block strictly before the evaluation window.
- **Option C:** A priori fixed configuration, documented before evaluation.

The choice must be declared before evaluation (invariant `III.1`). The evaluation window is never used for selection.

---

## 11. Missing Data Contract

### 11.1 Rules

- **Target missing:** Drop the sample. No imputation of `y_t`.
- **Feature missing, trailing:** Forward-fill is allowed only if the value was available at an earlier date. Documented per feature.
- **Feature missing, interior:** Drop the sample, or impute using a documented, past-only method.
- **Macro missing due to publication lag:** Use the last available vintage value; do not interpolate forward.
- **Asset missing over an extended period:** Exclude from the universe a priori, or move the start date forward.

### 11.2 Imputation constraints

If imputation is used:

- Only past data may be used.
- The imputation method must be deterministic.
- The imputation rate must be reported per feature.
- Any feature with imputation rate > 5% must be flagged.

### 11.3 Non-imputation option

The default is **no imputation** for target, and **drop-if-missing** for features unless a specific feature has a documented forward-fill rule.

---

## 12. Reproducibility Contract

### 12.1 Determinism

- Given the same configuration and the same raw data, features and targets are identical.
- Random seeds are fixed and recorded (invariant `IV.2`).
- Data snapshot hashes are recorded.

### 12.2 Configuration

Each run records:

- Start date, end date
- Training window type and size
- Feature set and transformations
- Data pull timestamp
- Random seed
- Package versions

### 12.3 Data snapshot

Each run stores:

- The raw data used
- The processed data produced
- The hash of both

If the raw data cannot be regenerated (e.g., vendor no longer serves it), the run's results are considered non-reproducible and this is disclosed.

### 12.4 Feature provenance

Each feature records the source series, transformations, and `available_at` timestamp. This is stored in a feature manifest alongside the processed data.

---

## 13. Data Contract for the Decision Rule

The decision rule (`05`) needs two inputs from the model:

- `μ_t = E[r_t | F_t]` — predictive mean
- `Σ_t = Cov[r_t | F_t]` — predictive covariance

The data contract ensures both are well-defined:

- `μ_t` is produced only from features in `F_t`.
- `Σ_t` is either produced by the model or estimated from `F_t` under a documented method.
- If `Σ_t` is estimated from historical returns, only data with reference date `< t` is used.
- No future data enters either statistic.

---

## 14. Ties to Other Documents

| Concern | Where specified |
|---|---|
| Utility function consuming `μ_t`, `Σ_t` | `05` |
| Decision rule | `05` |
| Constraints | `05`, Section 6 |
| Costs | `05`, Section 7 |
| Evaluation protocol | `07` |
| Architecture for data pipeline | `08` |
| Assumptions behind data choices | `04`, Section 4 |
| Scope of universe | `03`, Section 4.1 |

Conflicts are resolved by updating the affected document, not by silent override.

---

## 15. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Universe is fixed before evaluation. | Prevents survivorship-like adjustments. |
| D2 | VIX is a feature only, never an asset. | Not tradable (`03`). |
| D3 | Cash is the risk-free proxy; explicit vs residual treatment declared a priori. | Two options documented in `05`. |
| D4 | Returns use total-return adjusted prices. | Dividends materially affect factor returns. |
| D5 | Macro features must use ALFRED vintages or be lagged conservatively. | Prevents look-ahead (`I.3`). |
| D6 | Lagged returns default `L = 12`. | Standard, matches v1, avoids proliferation. |
| D7 | Feature construction obeys R1–R7. | Comprehensive leakage prevention. |
| D8 | Target is next-month total return, 1-month horizon. | Matches rebalancing. |
| D9 | No target imputation. | Any imputation biases results. |
| D10 | Feature imputation only under documented forward-fill. | Conservative and auditable. |
| D11 | Validation checks V1–V21 are mandatory. | Makes the contract machine-checkable. |
| D12 | Walk-forward only. | Invariant `I.6`. |
| D13 | Model selection never uses the evaluation window. | Invariant `III.1`. |
| D14 | Provenance is recorded for every series. | Reproducibility (`IV.3`). |

---

## 16. Open Questions

1. Should the universe include `CASH` as a formal asset in the return vector, or only as a residual? Affects covariance dimension and allocation semantics.
2. Should the macro feature set be finalized now or after MVP modeling? Current default: deferred to `08`, with the initial list in Section 6.2 as non-final.
3. Is the `T / p ≥ 5` rule appropriate for a walk-forward setting where `T` grows, or should the binding constraint be the smallest training window?
4. Should `VIXCLS` be included in the macro feature set given `B5` (fat tails) and its known regime-sensitivity?
5. Is a 1-month horizon consistent with the quarterly frequency of some macro data, or does it waste information? Alternative: use a longer horizon.
6. Should the target be excess returns (relative to cash) or raw total returns? Excess returns simplify utility in mean-variance form but require careful cash accounting.
7. What is the correct behavior if ALFRED vintage data is unavailable for a specific series on a specific date? Current default: drop the feature for that date, forward-fill the last available vintage value, and record it.
8. Should validation check V13 (correlation ceiling) abort the run or warn? Current default: warn, because a high correlation may reflect a genuine predictor rather than leakage. But the default ceiling of 0.95 may be too high.
9. Should features be winsorized or z-scored before modeling, and if so, using what window? Deferred to `08`.
10. Should the pipeline enforce that every feature's `available_at` timestamp is exactly the last business day of the prior month, or allow intra-month availability? Current default: exact month-end to align with the rebalancing calendar.
11. Should the data contract define a single canonical monthly calendar (e.g., NYSE trading days), or accept the union of trading days across assets?
12. What is the correct handling when a newly listed ETF has less than `L` months of history at the start of the evaluation window?

---

## 17. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial data, information, and target contract for v2. |

---

## 18. References

- `00_docs_index.md` — documentation map and conventions.
- `01_first_principles_problem_decomposition.md` — primitive decomposition.
- `02_problem_framing.md` — formal problem statement.
- `03_scope_and_non_goals.md` — scope boundaries.
- `04_assumptions_and_invariants.md` — assumptions catalog and invariant set.
- `05_objective_utility_and_decision_specification.md` — decision problem.
- ALFRED — Federal Reserve Bank of St. Louis vintage data.
- FRED — Federal Reserve Economic Data.
- yfinance — market data retrieval.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (For data leakage and split discipline.)
