# Evaluation Framework and Backtesting Protocol

**Project:** moe-factor-timing v2
**Status:** Draft
**Last Updated:** 2026-08-03

---

## 1. Purpose

This document defines **how the project measures success**, **how it backtests**, and **what claims it is allowed to make**.

Its job is to prevent the three evaluation failures of v1:

1. **Selection bias.** v1 used the same backtest to select `K = 4`, `min_train = 96`, `100 EM iterations`, and `magnitude weighting` that it then used to claim outperformance. Any result produced by this process is invalid regardless of its Sharpe ratio.
2. **Underpowered inference.** v1's 42-month out-of-sample window was treated as evidence of skill. It is not.
3. **Metric shopping.** v1 reported Sharpe ratio as the headline metric while its RMSE was worse than baselines. The trade-off was acknowledged but not formalized.

This document fixes those failures by:

- Specifying the evaluation protocol **before** any model is evaluated.
- Enforcing a strict separation between model selection and evaluation.
- Requiring uncertainty quantification for every reported metric.
- Requiring multiple-testing correction when more than one configuration is tested.
- Defining a reporting standard that prevents cherry-picking.

This document is a **contract**. Results produced under a different protocol are not reportable.

---

## 2. Context and Dependencies

- **Depends on:** `00_docs_index.md`, `01_first_principles_problem_decomposition.md`, `02_problem_framing.md`, `03_scope_and_non_goals.md`, `04_assumptions_and_invariants.md`, `05_objective_utility_and_decision_specification.md`, `06_data_information_and_target_contract.md`
- **Feeds into:** `08_minimum_viable_baseline_end_to_end_pipeline_architecture.md`, `09_stop_criteria_and_kill_criteria.md`
- **Reference only:** v1 `problem_framing.md`, v1 `README.md`, v1 `TODO.md`

The decision problem (`05`) and the data contract (`06`) define what can be evaluated. This document defines how.

---

## 3. Evaluation Philosophy

Four principles govern evaluation.

### 3.1 Decision-first

Predictive accuracy is a means, not an end. Metrics are ranked by how directly they measure decision quality. RMSE and MAE are reported as diagnostics, not as objectives.

### 3.2 Uncertainty-first

Every reported number has uncertainty. A Sharpe ratio of 1.49 over 42 months is compatible with a true Sharpe of 0. A Sharpe ratio of −0.5 over 42 months is compatible with a true Sharpe of +0.5. Both must be reported with confidence intervals.

### 3.3 Robustness-first

A result that holds only in one configuration is not evidence. Every headline result is reported across at least three robustness dimensions.

### 3.4 Honesty-first

Negative results are results. A benchmark that fails to find an edge is a successful benchmark. A benchmark that claims an edge it cannot support is a failure.

---

## 4. Evaluation Layers

Evaluation is organized in five layers. Each layer answers a distinct question.

| Layer | Question | Metrics | Section |
|-------|----------|---------|---------|
| L1 | Is the predictive distribution well-calibrated? | NLL, CRPS, PIT, calibration plots | 5 |
| L2 | Is the point forecast accurate (as a diagnostic)? | RMSE, MAE, R² | 6 |
| L3 | Are the allocations decision-useful after costs? | Utility, Sharpe, Sortino, MaxDD, Calmar, turnover, capacity | 7 |
| L4 | Is the performance distinguishable from chance? | Bootstrap CIs, DM, SPA, deflated Sharpe | 8 |
| L5 | Is the performance robust? | Multi-window, multi-seed, multi-config, multi-cost | 9 |

A model is only declared "useful" if it passes L1, L3, L4, and L5. L2 is informative but not decisive.

---

## 5. Layer 1 — Predictive Distribution Metrics

These metrics evaluate the probabilistic forecasts produced by the model. They are the primary predictive metrics.

### 5.1 Negative Log-Likelihood (NLL)

```
NLL = − (1/T) Σ_t log p̂( y_t | F_t )
```

where `p̂( y_t | F_t )` is the predictive density evaluated at the realized target.

**Interpretation:** Lower is better. Rewards both accuracy and calibration.

**Note:** NLL requires a probabilistic forecast. Models that only produce point forecasts are evaluated under L2 only and excluded from L1.

### 5.2 Continuous Ranked Probability Score (CRPS)

```
CRPS = (1/T) Σ_t ∫ ( F̂_t(z) − 1{z ≥ y_t} )² dz
```

**Interpretation:** Lower is better. Proper scoring rule. Comparable across models without distributional assumptions.

### 5.3 Probability Integral Transform (PIT)

For each observation, compute:

```
u_t = F̂_t( y_t )
```

Under a perfectly calibrated forecast, `u_t` is uniform on `[0, 1]`.

**Diagnostics:**
- PIT histogram (uniformity test)
- Kolmogorov-Smirnov test for uniformity
- Anderson-Darling test for tails

**Interpretation:** Deviation from uniformity indicates miscalibration. Overconfidence produces a U-shaped histogram. Underconfidence produces a dome-shaped histogram.

### 5.4 Calibration plots

For each quantile level `q ∈ {0.05, 0.10, ..., 0.95}`:

- Nominal coverage `q`
- Empirical coverage = fraction of `y_t` with `F̂_t( y_t ) ≤ q`

A well-calibrated forecast has empirical coverage ≈ nominal coverage.

**Plot:** Nominal vs empirical. Diagonal is perfect calibration.

### 5.5 Interval score

For a central `(1 − α)` prediction interval `[l_t, u_t]`:

```
IS_α = (u_t − l_t) + (2/α) · (l_t − y_t)·1{y_t < l_t} + (2/α) · (y_t − u_t)·1{y_t > u_t}
```

Average over `t` and report for `α ∈ {0.10, 0.20}`.

### 5.6 Primary predictive metric

**Primary:** NLL.

**Rationale:** It is the only metric that evaluates the full predictive distribution, which is the object consumed by the decision rule (`05`, Section 5.4). RMSE and MAE are secondary diagnostics.

---

## 6. Layer 2 — Point Forecast Diagnostics

These metrics evaluate the point predictions. They are **diagnostics**, not objectives. They are reported to characterize model behavior and to compare against the v1 results.

### 6.1 Metrics

- **RMSE:** `sqrt( (1/T) Σ_t || y_t − ŷ_t ||² )`
- **MAE:** `(1/T) Σ_t || y_t − ŷ_t ||₁`
- **R²:** `1 − SSE / SST` per asset

### 6.2 Reporting rules

- RMSE and MAE are always reported **per asset** and **aggregated**.
- Aggregated RMSE uses the Euclidean norm, not the mean of per-asset RMSEs.
- Point forecasts are always compared against the naive baselines in Section 11.

### 6.3 Why these are diagnostics

A model with low RMSE but poor decision performance is not decision-useful. A model with high RMSE but good decision performance may be. The relationship between RMSE and decision quality is not assumed.

---

## 7. Layer 3 — Decision Metrics

These metrics evaluate the actual portfolio outcomes. They are the **primary metrics** for the project.

### 7.1 Definitions

Let `w_t*` be the chosen weights at time `t`, `r_t` the realized returns, and `C_t` the transaction cost at time `t`.

Net portfolio return:

```
R_t = w_t*^T r_t − C_t
```

Cumulative net return:

```
W_T = Π_{t=1}^T ( 1 + R_t )
```

### 7.2 Core decision metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Expected utility** | `(1/T) Σ_t U( w_t*^T r_t )` | Primary. Matches the objective in `05`. |
| **Annualized return** | `W_T^{12/T} − 1` | Net of costs. |
| **Annualized volatility** | `sqrt(12) · std(R_t)` | |
| **Sharpe ratio** | `( mean(R_t) / std(R_t) ) · sqrt(12)` | Standard, but incomplete. |
| **Sortino ratio** | Uses downside deviation | Robustness metric. |
| **Maximum drawdown** | `max_t ( 1 − W_t / max_{s ≤ t} W_s )` | Path-based risk. |
| **Calmar ratio** | `annualized_return / |max_drawdown|` | |
| **Win rate** | Fraction of months with `R_t > 0` | |
| **Skewness** | Third standardized moment of `R_t` | Tail asymmetry. |
| **Excess kurtosis** | Fourth standardized moment minus 3 | Tail heaviness. |

### 7.3 Cost and turnover metrics

| Metric | Definition |
|--------|------------|
| **Turnover** | `(1/2) Σ_t ||w_t* − w_{t-1}*||₁ / T` |
| **Total costs** | `Σ_t C_t` |
| **Cost drag** | `Σ_t C_t / W_T` |
| **Hit rate on rebalances** | Fraction of rebalances where the trade improved realized return |

Turnover is reported with every headline metric (invariant `II.4`).

### 7.4 Capacity metric

Define capacity as the AUM level at which modeled market impact consumes a specified fraction of gross return (default 25%).

Capacity is estimated using a documented impact model (e.g., square-root impact with a specified coefficient). The impact model is documented in `08_...`.

Capacity is reported as a single number (in USD) with the model assumption stated.

### 7.5 Primary decision metric

**Primary:** Expected utility under the default utility function (`05`, Section 4.3).

**Rationale:** It is the object defined by the objective. Sharpe, Sortino, MaxDD, and Calmar are reported as complements, not substitutes.

**Reporting rule:** If the utility ranking differs from the Sharpe ranking, both are reported and the divergence is discussed.

### 7.6 Benchmarks in decision space

Every model is compared against the following benchmarks:

| Benchmark | Definition |
|-----------|------------|
| **1/N** | Equal weight across assets each month. |
| **Cash** | 100% in `CASH`. |
| **Persistence** | Next month's return = current month's return. |
| **Rolling average** | Mean of last 12 months. |
| **Momentum** | EWMA of past returns, decay = 0.9. |

Benchmarks are computed under the same cost model, constraints, and rebalancing calendar as the models. Comparing a net-of-cost model against a gross benchmark is prohibited.

---

## 8. Layer 4 — Statistical Validation

This layer establishes whether observed performance can be distinguished from chance.

### 8.1 Confidence intervals

For every headline metric, compute a bootstrap confidence interval:

- **Method:** Stationary block bootstrap (block length 3–6 months) to preserve autocorrelation.
- **Repetitions:** 1,000 minimum.
- **Level:** 95%.
- **Deterministic seed:** Recorded.

Reporting a point estimate without a confidence interval is prohibited.

### 8.2 Diebold–Mariano test

For pairwise comparison of forecast accuracy between two models:

```
DM = d̄ / sqrt( 2π f̂_d(0) / T )
```

where `d_t` is the loss differential at time `t` (using the primary predictive loss), `f̂_d(0)` is the spectral density at frequency 0, and `T` is the sample size.

**Use:** Pairwise comparison of NLL or CRPS across models.

**Adjustment:** Newey–West HAC for autocorrelation.

### 8.3 Hansen SPA test

For multiple-model comparison against a benchmark:

- Null: No model beats the benchmark after accounting for multiple testing.
- Statistic: Studentized maximum of performance differentials.
- Bootstrap: Stationary bootstrap.
- Level: 5%.

**Use:** When more than two models are compared, use SPA, not multiple pairwise DM tests.

### 8.4 Deflated Sharpe ratio

Adjusts the observed Sharpe ratio for:

1. Number of trials (`N_trials`)
2. Non-normality of returns
3. Length of sample

Formula (Bailey & López de Prado, 2014):

```
DSR = Φ( ( SR_obs − SR_expected ) · sqrt(T − 1) / sqrt( 1 − γ_3 · SR_obs + (γ_4 − 1)/4 · SR_obs² ) )
```

where `SR_expected` is the expected maximum Sharpe from `N_trials` independent trials.

**Use:** Required whenever more than one configuration is tested.

**Disclosure:** `N_trials` must be reported alongside the DSR.

### 8.5 Minimum track record length

Compute the minimum sample length required for the observed Sharpe to be significant at 95%:

```
MinTRL = 1 + ( 1 − γ_3 · SR + (γ_4 − 1)/4 · SR² ) · ( z_{0.95} / (SR − SR_benchmark) )²
```

**Use:** Compare `MinTRL` to `T`. If `MinTRL > T`, the result is underpowered.

### 8.6 Multiple-testing accounting

Whenever more than one configuration is tested, the following must be reported:

- Total number of configurations tested (`N_trials`)
- Search space dimensions
- Selection criterion
- Multiple-testing correction applied

Failing to report `N_trials` invalidates any significance claim.

### 8.7 Power analysis

Before evaluation, compute the minimum detectable Sharpe ratio at 95% confidence given the planned sample size `T`. If the minimum detectable Sharpe is larger than a reasonable expected effect, the evaluation is underpowered and this must be disclosed in advance.

**Rule:** If minimum detectable Sharpe > 1.0 for the planned `T`, the evaluation cannot support Sharpe-based claims and the project relies on utility-based and calibration-based claims instead.

---

## 9. Layer 5 — Robustness

Robustness is required for any headline result (invariant `III.6`).

### 9.1 Robustness dimensions

Every headline result is reported across:

| Dimension | Levels | Purpose |
|-----------|--------|---------|
| **Training window** | `min_train ∈ {60, 96, 132}` or rolling vs expanding | Stability. |
| **Number of latent states** | `K ∈ {1, 2, 3, 4, 5, 6}` | Model selection sensitivity. |
| **Cost assumption** | `{5, 10, 20} bps` assumed; calibrated | Cost sensitivity. |
| **Utility function** | Mean-variance; CRRA; CVaR | Objective sensitivity. |
| **Risk aversion** | `λ ∈ {2, 5, 10, 20}` | Preference sensitivity. |
| **Random seed** | 5 seeds minimum for stochastic models | Reproducibility. |
| **Evaluation subperiod** | Contiguous 12-month blocks | Time stability. |
| **Universe composition** | Drop each asset in turn | Asset sensitivity. |
| **Feature set** | VIX-only vs. full | Feature sensitivity. |

### 9.2 Reporting rule

For each headline result, report the **distribution across robustness dimensions**, not just the best.

- **Median** across dimensions
- **Interquartile range**
- **Minimum and maximum**

If a result only holds under one configuration, this is disclosed explicitly.

### 9.3 Robustness failure

If the sign of the primary metric flips across robustness dimensions, the result is not robust. It is reported as "not robust" and cannot serve as a headline claim.

---

## 10. Backtesting Protocol

### 10.1 Walk-forward structure

- **Window type:** Expanding (default) or rolling (variant).
- **Minimum training size:** 96 months (default), 60 and 132 as robustness.
- **Test step:** 1 month.
- **Rebalancing:** Monthly.
- **No shuffling.**
- **No cross-split leakage.**

### 10.2 Per-step protocol

At each test time `t`:

1. Fit all models on data ending at `t-1`.
2. Produce predictive distributions on `F_t`.
3. Solve the decision problem to obtain `w_t*`.
4. Assert constraints.
5. Record `w_t*`, solver status, predictive distribution, and diagnostics.
6. After observing `r_t`, compute `R_t = w_t*^T r_t − C_t`.
7. Update wealth.
8. Advance to `t+1`.

### 10.3 Cost handling

- Costs are deducted at every rebalance, inside the backtest.
- Costs are computed from `w_t* − w_{t-1}*`.
- Costs are never applied post hoc.

### 10.4 Cold start

- At the first rebalance, `w_{t-1}` is initialized to cash (or `1/N`, documented).
- The first month's cost includes the cost of moving from the initial allocation to `w_t*`.

### 10.5 Rebalance failures

- If the solver fails, the fallback (`w_t = w_{t-1}`) is applied.
- Fallback events are recorded.
- If fallback frequency exceeds 5%, the run is invalid (`05`, Section 9.3).

### 10.6 Model selection separation

- Hyperparameters are selected on a **separate validation window** or via **nested cross-validation inside the training window**.
- The evaluation window is never used for model selection.
- Violating this rule invalidates the run.

### 10.7 Freezing the protocol

- Before any evaluation runs, the following are frozen:
  - Universe
  - Features
  - Benchmarks
  - Utility function
  - Risk aversion grid
  - Cost scenarios
  - Robustness dimensions
  - Primary metric
  - Multiple-testing correction method
  - Reporting standard

- Any change requires restarting the evaluation from scratch.

### 10.8 Pre-registration

The frozen protocol is committed to version control before evaluation. The commit hash is included in the final report.

---

## 11. Benchmarks

Benchmarks are declared in `Section 7.6` and are frozen before evaluation (invariant `III.3`).

### 11.1 Benchmark rules

- Benchmarks use the same cost model, constraints, and calendar as models.
- Benchmarks do not use the same decision rule unless specified (e.g., 1/N is a fixed allocation rule).
- Benchmarks are reported in every table, every figure, and every summary.

### 11.2 Benchmark set

- **1/N** — Trivial baseline.
- **Cash** — Minimum risk baseline.
- **Persistence** — Naive time-series baseline.
- **Rolling average** — Smoothed baseline.
- **Momentum** — Trend-following baseline.

No benchmark may be added after evaluation begins unless the protocol is restarted.

---

## 12. Reporting Standard

### 12.1 Every reported result includes

- Point estimate
- Confidence interval (bootstrap)
- Sample size
- Number of configurations tested
- Multiple-testing correction applied
- Robustness distribution across Section 9.1 dimensions
- Turnover
- Costs (in bps and in drag)
- Capacity estimate
- Number of rebalance failures

A result missing any of these is not reportable.

### 12.2 Every claim is qualified

- "Outperformed" is not allowed without specifying the metric, the sample, the confidence level, and the correction.
- "Significant" is not allowed without a specific test and a reported p-value or DSR.
- "Robust" is not allowed without the robustness distribution.

### 12.3 No selection of the best window

If multiple windows are evaluated, the full distribution is reported. Reporting only the best window is prohibited.

### 12.4 No cherry-picked figures

If multiple figures are produced, all are stored in `results/` with a manifest. Reporting only selected figures is prohibited.

### 12.5 Negative results

Negative results are reported with the same rigor as positive results. A failure to find an edge is a finding.

### 12.6 Disclaimer

Every reported result is accompanied by the disclaimer that backtest performance does not imply live performance (`04`, `F5`).

---

## 13. Reproducibility

Reproducibility follows from `06`, Section 12, plus the following additions.

### 13.1 Run manifest

Each run produces a manifest containing:

- Configuration hash
- Data snapshot hash
- Code commit hash
- Package versions
- Random seeds
- Start and end timestamps
- Output paths

### 13.2 Config freeze

The configuration used for evaluation is frozen and stored with the run. Changes require a new run ID.

### 13.3 Result hashes

All output artifacts (CSVs, figures, metrics) are hashed and recorded. Reproduction with the same inputs must produce identical hashes.

### 13.4 Environment capture

Python version, OS, and package versions are captured. Differences in environment that could affect floating-point results are disclosed.

---

## 14. Regime Analysis Evaluation

Regime-aware models produce latent state assignments. These are evaluated separately from decision performance.

### 14.1 Regime metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Frequency** | Fraction of periods assigned to each state | Balance check |
| **Persistence** | Average run length per state | Stability |
| **Transition matrix** | Empirical transitions between states | Markov property check |
| **Conditional mean** | Mean return of assets conditional on state | Economic content |
| **Conditional volatility** | Std of returns conditional on state | Risk content |
| **Conditional correlation** | Pairwise correlations conditional on state | Diversification content |
| **Entropy** | Entropy of the posterior over states | Confidence |

### 14.2 Regime stability tests

- **Window stability:** Assignments in `[t, t+12]` vs. `[t+12, t+24]` — compare via mutual information.
- **Seed stability:** Assignments across seeds — compare via adjusted Rand index.
- **Hyperparameter stability:** Assignments across `K` values — interpretability check.

### 14.3 Regime economic interpretability

- Overlay states on known macro dates (recessions, crisis periods).
- Correlate state probabilities with macro indicators.
- Report whether states correspond to economically recognizable conditions.

**Rule:** States are not labeled "recession", "risk-on", etc. unless there is direct empirical support. Descriptive labels ("state 1") are the default.

### 14.4 Regime claims

No claim that "regimes exist" or "the model identified regimes" is permitted. The claim is limited to: "the model produced latent state assignments with the following empirical characteristics."

---

## 15. Failure Modes and Detection

| Failure | Detection | Response |
|---------|-----------|----------|
| Look-ahead bias | Validation checks V8–V11 | Abort run |
| Target leakage | V13, V14 | Investigate, likely abort |
| Overfitting via selection | Protocol audit | Abort, restart |
| Multiple testing without correction | Reporting audit | Add correction, restart |
| Unstable solver | Fallback frequency | Abort if > 5% |
| Non-convergent training | Training diagnostics | Report, may abort |
| Underpowered inference | Power analysis (`8.7`) | Report as underpowered |
| Cherry-picked window | Reporting audit | Report all windows |
| Metric shopping | Reporting audit | Report all metrics |
| Silent scope change | Version control audit | Restart |

---

## 16. Ties to Other Documents

| Concern | Where specified |
|---|---|
| Objective and utility | `05` |
| Constraint set | `05`, Section 6 |
| Cost model | `05`, Section 7 |
| Data contract | `06` |
| Invariants | `04`, Section 10 |
| Assumptions behind evaluation | `04`, Section 9 |
| Stop and kill criteria | `09` |
| Architecture of evaluation module | `08` |

Conflicts are resolved by updating the affected document, not by silent override.

---

## 17. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Primary predictive metric is NLL. | Only metric on full distribution. |
| D2 | Primary decision metric is expected utility. | Matches the objective. |
| D3 | RMSE and MAE are diagnostics only. | Not decision-relevant on their own. |
| D4 | Every headline result has a bootstrap CI. | Point estimates mislead. |
| D5 | Multiple-testing correction is mandatory. | Prevents selection bias. |
| D6 | Deflated Sharpe ratio is required when `N_trials > 1`. | Explicit selection accounting. |
| D7 | Minimum track record length is reported. | Reveals underpowered claims. |
| D8 | Power analysis is done before evaluation. | Prevents overclaiming. |
| D9 | Robustness across 9 dimensions is required. | Single-window results are not evidence. |
| D10 | Evaluation protocol is frozen before evaluation. | Prevents goalpost-moving. |
| D11 | Model selection uses a separate window. | Invariant `III.1`. |
| D12 | Benchmarks are declared before evaluation. | Invariant `III.3`. |
| D13 | Reporting standard is mandatory. | Prevents cherry-picking. |
| D14 | Negative results are reported with the same rigor. | Honesty-first principle. |
| D15 | Regime claims are descriptive, not ontological. | Prevents overclaiming. |
| D16 | Run manifest and result hashes are required. | Reproducibility. |

---

## 18. Open Questions

1. Is NLL well-defined for models that only produce mixtures with estimated covariances? If the covariance estimate is rank-deficient, NLL may be infinite. Fallback?
2. Is the stationary block bootstrap appropriate given that walk-forward predictions are not stationary by construction?
3. Should the primary metric be raw expected utility or utility per unit of risk? Utility scales with wealth and risk aversion, making cross-configuration comparison non-trivial.
4. Should the SPA test be the default multiple-testing correction, or should the deflated Sharpe be the default? They answer different questions.
5. Is a minimum detectable Sharpe of 1.0 the correct threshold for declaring underpowering, or should it be higher given the small sample?
6. Should capacity be reported as a single AUM number, a curve, or both?
7. Should robustness results be reported in the paper or only in supplementary material?
8. How should conflicting robustness results be resolved (e.g., metric positive in 5 of 9 dimensions)?
9. Is the 5% fallback threshold for solver failures consistent with the 42-month minimum evaluation window, or should it be tighter?
10. Should the deflated Sharpe's `N_trials` count every model hyperparameter combination, or only every distinct model-family choice?
11. Should PIT calibration be reported per asset or only aggregated?
12. Is the current regime analysis sufficient to support descriptive claims, or does it also require a null-model comparison (e.g., random assignment with the same frequency)?
13. Should the paper report the naive model results prominently, or only in an appendix, given that v1 reported naive models as "outperformed"?
14. Is the current frozen-protocol requirement compatible with iterative development, or does it imply a complete code freeze before any model is run?
15. How should the project respond if the primary metric is insignificant but a secondary metric is significant? Current answer: report both, primary takes precedence, no claim of success.

---

## 19. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial evaluation framework and backtesting protocol for v2. |

---

## 20. References

- `00_docs_index.md` — documentation map and conventions.
- `01_first_principles_problem_decomposition.md` — primitive decomposition.
- `02_problem_framing.md` — formal problem statement.
- `03_scope_and_non_goals.md` — scope boundaries.
- `04_assumptions_and_invariants.md` — assumptions and invariants.
- `05_objective_utility_and_decision_specification.md` — objective and decision rule.
- `06_data_information_and_target_contract.md` — data contract.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics*, 23(4), 365–380.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5), 94–107.
- Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
