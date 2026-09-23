# Assumptions and Invariants

**Project:** moe-factor-timing v2
**Status:** Draft
**Last Updated:** 2026-08-03

---

## 1. Purpose

This document does two things:

1. It **catalogs every assumption** the project depends on, and classifies each one as **accepted**, **rejected**, or **hypothesis to test**.
2. It defines **invariants** — conditions that must hold at every point in the pipeline, regardless of model, configuration, or experiment.

Its job is to make hidden assumptions visible and to make non-negotiable rules explicit.

The failure mode this document prevents is the one that plagued v1: assumptions made implicitly, never tested, and never separated from facts. In v1, choices like "K=4", "96 months", "magnitude weighting", and "10 bps" were treated as if they were properties of the problem. They were not. They were assumptions.

A hidden assumption is a bug. An invariant is a guardrail.

---

## 2. Context and Dependencies

- **Depends on:** `00_docs_index.md`, `01_first_principles_problem_decomposition.md`, `02_problem_framing.md`, `03_scope_and_non_goals.md`
- **Feeds into:** `05_objective_utility_and_decision_specification.md`, `06_data_information_and_target_contract.md`, `07_evaluation_framework_and_backtesting_protocol.md`
- **Reference only:** v1 `problem_framing.md`, v1 `README.md`, v1 `TODO.md`

Every assumption in this document must be either:

- **Accepted** — with a written justification.
- **Rejected** — with a written reason.
- **Hypothesis to test** — with a corresponding research question in `02_problem_framing.md` and a test in `07_...`.

No assumption may remain unclassified. Silence is not a classification.

---

## 3. Taxonomy

Assumptions are grouped by the sub-problem they affect:

- **A — Data and information assumptions**
- **B — Market and return assumptions**
- **C — Model and inference assumptions**
- **D — Decision and utility assumptions**
- **E — Cost and execution assumptions**
- **F — Evaluation and statistical assumptions**

Invariants are grouped by the pipeline stage they guard:

- **I — Information set invariants**
- **II — Decision invariants**
- **III — Evaluation invariants**
- **IV — Reproducibility invariants**

Each assumption gets a stable identifier (`A1`, `A2`, ...). Each invariant gets a stable identifier (`I.1`, `II.3`, ...). Identifiers are never reused, even if an item is rejected.

---

## 4. Data and Information Assumptions

### A1 — Historical price data is accurate and complete

**Classification:** Accepted, with caveats.

**Justification:** We rely on standard vendors (yfinance, FRED, ALFRED). We do not claim vendor data is perfect. We document the vendor, retrieval timestamp, and any known issues.

**Caveats:** Missing values, dividend adjustments, and survivorship bias in ETF histories are known limitations. We do not implement our own data cleaning beyond documenting gaps.

### A2 — Macroeconomic indicators are published with a lag

**Classification:** Accepted.

**Justification:** CPI, INDPRO, UNRATE, and similar series are revised and released with delays. Using them at their reference date is look-ahead bias.

**Consequence:** We require vintage-aware data (ALFRED or equivalent) for any macro feature. See invariant I.3.

### A3 — ETF prices reflect investable returns

**Classification:** Hypothesis to test.

**Justification:** ETFs are tradable, but they have tracking error, fees, and may not perfectly replicate the underlying factor.

**Test:** Compare ETF returns against academic factor series where available. Report tracking error as a limitation.

### A4 — Monthly frequency captures the relevant signal

**Classification:** Accepted for v2.0, deferred for testing.

**Justification:** Macro indicators are released monthly. Higher frequencies would be dominated by microstructure noise at the macro level.

**Deferred test:** Whether quarterly rebalancing changes conclusions materially.

### A5 — Missing data can be handled by dropping or imputing

**Classification:** Accepted with a rule.

**Rule:** If a required series has missing values at time `t`, the observation is dropped from the training set. Imputation is allowed only for non-target features, using only past information, and must be documented.

**Justification:** Dropping is conservative. Imputation is convenient but introduces bias if done carelessly.

### A6 — The information set `F_t` is fully representable

**Classification:** Accepted for v2.0, with known limitation.

**Justification:** We cannot represent everything a market participant knows. We represent a specific, documented subset.

**Limitation:** The project's claims are conditional on this representation. If key information is omitted, the results are conditional on that omission.

---

## 5. Market and Return Assumptions

### B1 — Returns are non-stationary

**Classification:** Accepted.

**Justification:** Empirical observation. This is a primitive in `01_first_principles_problem_decomposition.md` (Section 5.6).

**Consequence:** No model may assume a fixed data-generating process. All evaluation must be walk-forward.

### B2 — The relationship between information and returns is not constant

**Classification:** Accepted.

**Justification:** Implied by B1. If returns are non-stationary, their conditional dependence on information is also non-stationary.

**Consequence:** Feature importance and regime structure must be analyzed for stability, not assumed.

### B3 — Latent regimes exist and are stable enough to be modeled

**Classification:** Hypothesis to test.

**Justification:** This is the central modeling hypothesis of the project. It is not a fact. It must be tested.

**Test:** Compare models with `K = 1` (no regimes) against models with `K > 1`. If `K > 1` does not improve out-of-sample decisions after costs, the regime hypothesis is not supported.

### B4 — Returns are conditionally independent across time given the latent state

**Classification:** Hypothesis to test.

**Justification:** This is a common modeling assumption. If false, the model misspecifies autocorrelation.

**Test:** Residual autocorrelation analysis on out-of-sample predictions.

### B5 — The return distribution has fat tails

**Classification:** Accepted.

**Justification:** Empirical regularity across asset classes.

**Consequence:** Gaussian assumptions are a starting point, not a final model. Heavy-tailed distributions and CVaR-based metrics are preferred where feasible.

### B6 — Correlations between factors change over time

**Classification:** Accepted.

**Justification:** Correlations spike during crises. A static covariance model is misspecified.

**Consequence:** Covariance estimation must be time-varying. Regime-conditional covariance is one option; rolling window or EWMA is another.

### B7 — The assets in the universe are not perfectly correlated

**Classification:** Accepted.

**Justification:** If all assets were perfectly correlated, allocation would be trivial. The factor universe is chosen partly for its low average cross-correlation.

**Consequence:** If realized correlations approach 1 during evaluation, the allocation problem degenerates. This must be reported, not hidden.

### B8 — No single factor dominates all others persistently

**Classification:** Hypothesis to test.

**Justification:** If one factor dominated, allocation would be trivial. If no factor dominates, allocation is meaningful.

**Test:** Report factor-by-factor performance across the evaluation window.

---

## 6. Model and Inference Assumptions

### C1 — A parametric model can approximate the conditional return distribution

**Classification:** Accepted with caveats.

**Justification:** No finite model can represent an arbitrary distribution. We accept approximation error as a limitation.

**Caveat:** Model risk is real. Multiple model classes are compared, not one.

### C2 — Model parameters can be estimated from finite data

**Classification:** Accepted with caveats.

**Justification:** We have limited history. Parameter estimates have sampling error.

**Caveat:** Parameter uncertainty is a first-class concern. Methods that represent it (Bayesian, ensemble) are preferred where feasible.

### C3 — MoE with linear experts is expressive enough to be a useful benchmark

**Classification:** Hypothesis to test.

**Justification:** Linearity within regimes is a strong assumption. It may or may not be sufficient.

**Test:** Compare against non-linear models (RF, GBM) under the same decision rule.

### C4 — Soft regime assignment is more decision-useful than hard assignment

**Classification:** Hypothesis to test.

**Justification:** Soft assignment preserves uncertainty. Hard assignment loses it. Whether this matters for decisions is empirical.

**Test:** Compare soft MoE against a hard-assignment variant under the same protocol.

### C5 — The number of regimes `K` is a model selection problem

**Classification:** Accepted.

**Justification:** `K` is a hyperparameter, not a property of the world.

**Consequence:** `K` must be selected using out-of-sample criteria, not by maximizing the backtest Sharpe of the same model it is meant to validate.

### C6 — Model uncertainty is real and cannot be ignored

**Classification:** Accepted.

**Justification:** Multiple model classes can fit the same data. Picking one and ignoring the others overstates confidence.

**Consequence:** Results must be reported across model classes. Where feasible, model averaging or ensembles are used.

### C7 — The backtest is not a validation set

**Classification:** Accepted as an invariant.

**Justification:** If backtest performance is used to select models, the backtest is no longer out-of-sample.

**Consequence:** See invariant III.1. Model selection must use a separate protocol or nested cross-validation.

### C8 — Hyperparameter tuning can overfit if not controlled

**Classification:** Accepted.

**Justification:** Any hyperparameter searched on the evaluation window inflates apparent performance.

**Consequence:** Hyperparameters are either fixed a priori, tuned on a separate window, or reported with the number of configurations tested.

### C9 — Training data must precede test data in time

**Classification:** Accepted as an invariant.

**Justification:** Shuffling introduces look-ahead bias.

**Consequence:** See invariant I.1. Walk-forward only.

---

## 7. Decision and Utility Assumptions

### D1 — The investor has a well-defined utility function

**Classification:** Accepted with specification.

**Justification:** Without a utility function, "optimal allocation" is undefined.

**Specification:** The utility function is documented in `05_objective_utility_and_decision_specification.md`. It is not chosen ad hoc.

### D2 — Mean-variance is a useful approximation

**Classification:** Hypothesis to test.

**Justification:** Mean-variance is convenient but ignores skewness and tails.

**Test:** Compare mean-variance allocation against CVaR-based allocation out-of-sample.

### D3 — Risk aversion is constant over time

**Classification:** Accepted for v2.0, deferred for testing.

**Justification:** Time-varying risk aversion would add parameters without a clear signal.

**Deferred test:** Whether regimes exhibit different implied risk aversion.

### D4 — Long-only allocation is a valid initial constraint

**Classification:** Accepted for v2.0.

**Justification:** Simplifies the decision problem and matches many institutional mandates.

**Limitation:** If shorting would materially change conclusions, long-only results are conditional on this constraint.

### D5 — No leverage is a valid initial constraint

**Classification:** Accepted for v2.0.

**Justification:** Keeps the risk profile bounded and interpretable.

### D6 — The allocation decision is single-period

**Classification:** Accepted for v2.0.

**Justification:** Multi-period optimization introduces path dependence and requires a terminal utility specification.

**Deferred:** Multi-period optimization, if single-period results justify further work.

### D7 — The allocation rule must be derived from the objective

**Classification:** Accepted as an invariant.

**Justification:** Ad hoc rules (e.g., "magnitude-weighted long on positive predictions") may work but cannot be evaluated against optimality.

**Consequence:** See invariant II.1.

### D8 — The decision rule must be fixed before evaluation

**Classification:** Accepted as an invariant.

**Justification:** Changing the decision rule after seeing results is a form of overfitting.

**Consequence:** See invariant III.2.

---

## 8. Cost and Execution Assumptions

### E1 — Transaction costs are non-zero

**Classification:** Accepted.

**Justification:** Real trading has costs. Ignoring them invalidates results.

### E2 — The cost model is an approximation

**Classification:** Accepted.

**Justification:** We do not have access to proprietary execution data.

**Consequence:** Costs are reported as either "assumed" or "calibrated" and the distinction is visible in results.

### E3 — Costs scale with turnover

**Classification:** Accepted.

**Justification:** Higher turnover implies more trading, which implies more cost.

**Consequence:** Turnover is reported alongside every performance metric.

### E4 — Costs are constant across assets

**Classification:** Hypothesis to test.

**Justification:** Real costs vary by liquidity, spread, and instrument.

**Test:** Sensitivity analysis with asset-specific cost assumptions.

### E5 — Costs are constant over time

**Classification:** Hypothesis to test.

**Justification:** Costs spike during stress.

**Test:** Sensitivity analysis with regime-conditional cost scenarios.

### E6 — Market impact is negligible at the modeled size

**Classification:** Accepted with caveat.

**Justification:** We assume the strategy operates at a size where impact is small.

**Caveat:** Capacity is reported. If the modeled size is unrealistic, impact becomes material.

### E7 — Costs cannot be avoided by not trading

**Classification:** Accepted.

**Justification:** Not trading is a valid action, but it has an opportunity cost equal to the deviation from the target allocation.

**Consequence:** The cost model must include the cost of deviating from target, not only the cost of trading.

---

## 9. Evaluation and Statistical Assumptions

### F1 — Out-of-sample evaluation is the only valid form of evidence

**Classification:** Accepted.

**Justification:** In-sample fit is not evidence of generalization.

### F2 — A 42-month evaluation window is insufficient for statistical claims

**Classification:** Accepted.

**Justification:** v1's 42-month window cannot support Sharpe ratio significance tests at reasonable power.

**Consequence:** Either extend the sample or report results as descriptive, not inferential. See `07_...` for the exact policy.

### F3 — Multiple testing inflates apparent significance

**Classification:** Accepted.

**Justification:** Testing many configurations and reporting the best one produces selection bias.

**Consequence:** Multiple-testing corrections (e.g., deflated Sharpe, SPA) are required.

### F4 — Sharpe ratio is a useful but incomplete metric

**Classification:** Accepted.

**Justification:** Sharpe ignores tails, skewness, autocorrelation, and drawdown.

**Consequence:** Sharpe is reported alongside Sortino, MaxDD, Calmar, turnover, and capacity.

### F5 — Backtest returns are not live returns

**Classification:** Accepted.

**Justification:** Slippage, timing, and unmodeled frictions are always worse live.

**Consequence:** Every reported backtest metric is accompanied by an explicit disclaimer.

### F6 — The benchmark must be specified before evaluation

**Classification:** Accepted as an invariant.

**Justification:** Changing the benchmark after seeing results is cherry-picking.

**Consequence:** See invariant III.3.

### F7 — Robustness across windows is required for any performance claim

**Classification:** Accepted.

**Justification:** A result that depends on one specific window is not evidence of skill.

**Consequence:** Every headline result must be reported across at least three windows or subperiods.

---

## 10. Invariants

Invariants are conditions that must hold at every point in the pipeline. If an invariant is violated, the run is invalid and results are discarded.

### 10.1 Information set invariants

**I.1 — No look-ahead in training data.**
All training samples at time `t` use only information available at or before `t-1`. Shuffling is prohibited in time-series contexts.

**I.2 — No target leakage.**
No feature may be constructed using any part of the target variable, including transformed versions, and including cross-sectional aggregates that use same-period data.

**I.3 — Vintage-aware macro features.**
Every macro feature must be drawn from the vintage that was available at the time of the decision. If vintage data is unavailable, the feature is either dropped or lagged by a documented, conservative amount.

**I.4 — Lagged portfolio state.**
The current portfolio weight `w_{t-1}` is part of `F_t`, but `w_t` is not. Turnover and cost computations use `w_{t-1}` and the proposed `w_t` only.

**I.5 — No future information in feature engineering.**
Rolling statistics use only past data. Z-scores, ratios, and derived features respect the same rule.

**I.6 — Test period is strictly after training period.**
For every prediction at time `t`, the training window ends at or before `t-1`.

### 10.2 Decision invariants

**II.1 — The allocation rule is derived from the objective.**
The mapping from predictive distribution to weights must be documented as the solution to the optimization problem in `05_...`. No ad hoc rules.

**II.2 — Constraints are enforced.**
Long-only, fully-invested (or cash-allowed), turnover cap, and position limits are enforced at every rebalance. Violations invalidate the result.

**II.3 — Costs are deducted from realized returns.**
Reported returns are net of modeled transaction costs. Gross returns may be reported as a diagnostic but never as the headline metric.

**II.4 — Turnover is reported alongside every performance metric.**
A performance number without its turnover is incomplete.

**II.5 — Risk is controlled, not assumed away.**
The risk penalty in the objective is active. If `λ = 0`, this must be justified and reported.

### 10.3 Evaluation invariants

**III.1 — The backtest is not used for model selection.**
If a configuration is chosen by maximizing backtest performance, that backtest is no longer out-of-sample. Model selection uses a separate protocol (nested CV, separate validation window, or a priori fixed configuration).

**III.2 — The decision rule is frozen before evaluation.**
Once the evaluation window is defined, the decision rule cannot be changed. Any change requires restarting the evaluation.

**III.3 — The benchmark is specified before evaluation.**
Benchmarks (1/N, cash, momentum, etc.) are declared before results are observed.

**III.4 — Every reported metric has a confidence statement.**
Point estimates alone are not sufficient. Where sample size allows, bootstrap confidence intervals or equivalent are reported. Where it does not, the limitation is stated.

**III.5 — Multiple-testing correction is applied.**
If more than one configuration is tested and the best is reported, a multiple-testing correction is applied. The number of configurations tested is disclosed.

**III.6 — Robustness checks are reported.**
Every headline result is accompanied by results across at least three robustness dimensions (e.g., window length, `K`, cost assumption).

### 10.4 Reproducibility invariants

**IV.1 — Environment is captured.**
Each run records the Python version, package versions, and the exact configuration used.

**IV.2 — Random seeds are fixed and recorded.**
Any stochastic component (model initialization, bootstrap, sampling) uses a recorded seed.

**IV.3 — Data provenance is recorded.**
For each run, the source, retrieval date, and any transformations of the input data are recorded.

**IV.4 — Results are reproducible from configuration alone.**
Given the configuration and the raw data, the reported results can be regenerated.

**IV.5 — Configurations are versioned.**
Configuration files are stored with each run, not overwritten.

**IV.6 — No silent code changes.**
Any change to pipeline code that affects results requires a new run and a new configuration record.

---

## 11. Invariant Violations — Response Protocol

If an invariant is violated:

1. **Stop the run.**
2. **Document the violation** in the run log: which invariant, when, and why.
3. **Discard the affected results.** They are not valid.
4. **Fix the root cause.** Not the symptom.
5. **Re-run from a clean state.** Do not patch results.

Partial results from a run with an invariant violation are not reportable. Reporting them with a caveat is not acceptable.

---

## 12. Assumption Review Protocol

Assumptions are not permanent. They are reviewed at three checkpoints:

1. **Before implementation** — this document is finalized.
2. **Before reporting** — every accepted assumption is re-checked against the evidence.
3. **After reporting** — assumptions that were tested and failed are documented in the report as limitations.

Rejected assumptions are never reinstated without an explicit new justification and a version bump to this document.

Hypotheses to test must have a corresponding test in `07_evaluation_framework_and_backtesting_protocol.md`. If a hypothesis has no test, it is either accepted (with justification) or rejected.

---

## 13. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Every assumption is classified as accepted, rejected, or hypothesis to test. | Silence is not a classification. |
| D2 | Invariants are non-negotiable. Violation invalidates the run. | Partial validity is not validity. |
| D3 | Macro data must be vintage-aware or dropped. | Otherwise the backtest is invalid. |
| D4 | The backtest is never used for model selection. | Otherwise it is not out-of-sample. |
| D5 | Every result has a confidence statement. | Point estimates mislead. |
| D6 | Every headline result has robustness checks. | Single-window results are not evidence. |
| D7 | Assumptions are reviewed before and after reporting. | Assumptions can become stale. |
| D8 | Invariant violations require full re-run, not patching. | Patching hides bias. |

---

## 14. Open Questions

1. Is `A6` (information set is fully representable) a real assumption or a known limitation mislabeled?
2. Should `B3` (regimes exist) be a hypothesis to test, or is it better treated as a modeling language without an ontological claim?
3. Is `C4` (soft assignment is more useful) testable independently of other modeling choices?
4. Is `D3` (constant risk aversion) acceptable as an assumption, or does it silently drive results?
5. Should `E4` and `E5` (constant costs) be accepted with a caveat instead of hypotheses to test?
6. Is `F2` a limitation of the data or a limitation of the project? If the data could be extended, does the assumption remain?
7. Should invariants be enforced programmatically (assertions in code) or by process (review)?
8. Is there a case where violating an invariant is acceptable as long as it is disclosed? Current answer: no. Is that too strict?

---

## 15. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial assumptions catalog and invariant set for v2. |

---

## 16. References

- `00_docs_index.md` — documentation map and conventions.
- `01_first_principles_problem_decomposition.md` — primitive decomposition and v1 assumption audit.
- `02_problem_framing.md` — formal problem statement and research questions.
- `03_scope_and_non_goals.md` — scope boundaries and deferred items.
- v1 `problem_framing.md` — reference only.
- v1 `TODO.md` — reference for v1 bug history.
