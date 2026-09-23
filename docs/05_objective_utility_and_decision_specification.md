# Objective, Utility, and Decision Specification

**Project:** moe-factor-timing v2
**Status:** Draft
**Last Updated:** 2026-08-03

---

## 1. Purpose

This document specifies **what "good" means** for the project.

It defines:

1. The investor objective.
2. The utility function that formalizes that objective.
3. The decision rule that maps a predictive distribution into portfolio weights.
4. The constraint set that bounds the decision.
5. The cost model that penalizes action.
6. The risk penalty that penalizes exposure.

Its job is to prevent the failure mode of v1: allocating capital using an ad hoc rule ("magnitude-weighted long on positive predictions") that was not derived from any objective, and then evaluating that rule with metrics (Sharpe ratio) that were not chosen from first principles.

This document is the **contract** that downstream code must satisfy. Invariant `II.1` (from `04_assumptions_and_invariants.md`) requires that the allocation rule be traceable to this document. If it is not, the code is invalid regardless of its backtest performance.

---

## 2. Context and Dependencies

- **Depends on:** `00_docs_index.md`, `01_first_principles_problem_decomposition.md`, `02_problem_framing.md`, `03_scope_and_non_goals.md`, `04_assumptions_and_invariants.md`
- **Feeds into:** `06_data_information_and_target_contract.md`, `07_evaluation_framework_and_backtesting_protocol.md`, `08_minimum_viable_baseline_end_to_end_pipeline_architecture.md`
- **Reference only:** v1 `problem_framing.md`, v1 `README.md`

This document is the bridge between the problem statement (`02`) and the data contract (`06`). The decision problem determines what data is required, not the other way around.

---

## 3. The Objective

### 3.1 Plain-language statement

The investor wants to allocate capital across a set of assets at each rebalancing period in a way that produces good risk-adjusted outcomes over time, net of costs, using only information available at the time of the decision.

"Good" is not defined by the returns alone. It is defined by a utility function that reflects the investor's tolerance for risk, drawdown, and turnover.

### 3.2 Formal statement

At each rebalancing time `t`, given:

- Information set `F_t`
- Predictive distribution `p(r_t | F_t)` over next-period returns `r_t ∈ R^N`
- Previous weights `w_{t-1} ∈ R^N`
- Admissible set `W ⊆ R^N`
- Utility function `U : R → R`
- Cost function `C : R^N × R^N → R_+`
- Risk penalty `R : R^N → R_+`
- Risk aversion `λ ≥ 0`

The investor solves:

```
w_t* = argmax_{w ∈ W}   E[ U( w^T r_t ) | F_t ]   −   C( w, w_{t-1} )   −   λ · R( w )
```

Everything downstream — data, model, backtest — serves this problem.

### 3.3 What the objective is not

- It is **not** minimizing prediction error.
- It is **not** maximizing Sharpe ratio.
- It is **not** beating a benchmark.
- It is **not** achieving a target return.

These are consequences or diagnostics. They are not the objective.

---

## 4. Utility Function Specification

### 4.1 Design requirements

The utility function must satisfy:

1. **Monotonicity.** More wealth is preferred to less: `U'(W) > 0`.
2. **Risk aversion.** `U''(W) < 0`.
3. **Finiteness under heavy tails.** `E[U(W)]` must remain finite for fat-tailed return distributions (`B5` in `04_...`).
4. **Parameter efficiency.** It must be estimable with the available sample (`F2`).
5. **Interpretability.** Its parameters must map to decisions a real investor can make.
6. **Consistency with drawdown.** It must penalize paths that produce large drawdowns, not only moments.

### 4.2 Candidate utility functions

Four candidates are considered. Only one is the default; the others are tested as robustness variants.

#### 4.2.1 Mean-variance utility

```
U_MV(w^T r) = w^T μ − (λ / 2) · w^T Σ w
```

- **Pros:** Simple, closed-form when combined with Gaussian returns, widely used.
- **Cons:** Ignores skewness and tails. Violates requirement 3 under heavy tails (`B5`).

**Status:** Accepted as the **baseline utility** for v2.0. Tested against alternatives in `07_...`.

#### 4.2.2 CRRA utility (power utility)

```
U_CRRA(W) = ( W^(1−γ) ) / ( 1 − γ )      for γ ≠ 1
U_CRRA(W) = log(W)                        for γ = 1
```

- **Pros:** Grounded in expected utility theory. Sensitive to tails for high `γ`.
- **Cons:** Requires terminal wealth specification, which is awkward in a single-period model.

**Status:** Accepted as a **candidate variant** for robustness. Not the default because the single-period formulation makes terminal wealth specification arbitrary.

#### 4.2.3 Exponential utility (CARA)

```
U_CARA(W) = − exp(−α · W)
```

- **Pros:** Constant absolute risk aversion. Tractable with heavy-tailed returns.
- **Cons:** Absolute (not relative) risk aversion is unusual for equity allocation.

**Status:** Accepted as a **candidate variant**. Not the default.

#### 4.2.4 CVaR-constrained utility

```
U_CVaR(w^T r) = w^T μ − λ · CVaR_α( w^T r )
```

where `CVaR_α` is the Conditional Value-at-Risk at level `α` (e.g., `α = 0.05`).

- **Pros:** Directly penalizes tail risk. Addresses the requirement 3 failure of mean-variance.
- **Cons:** Requires a predictive distribution with enough tail samples. Requires convex optimization.

**Status:** Accepted as the **primary robustness variant**. Used for tail-risk robustness tests.

### 4.3 Default utility for v2.0

**Default:** Mean-variance utility (4.2.1).

**Rationale:**
- It matches the standard formulation used in the factor allocation literature.
- It is analytically tractable and matches the information available.
- It is a baseline against which the tail-aware alternatives (CRRA, CVaR) can be compared.

**Robustness variants:**
- Mean-variance vs. CRRA (with `γ ∈ {2, 5, 10}`).
- Mean-variance vs. CVaR (`α ∈ {0.05, 0.10}`).

**Reporting rule:** If results differ qualitatively across utility functions, all variants are reported. If they agree, only the default is reported, with a note that alternatives were tested.

### 4.4 Risk aversion parameter

`λ` is not arbitrary. It is either:

1. **Set a priori** using a documented calibration (e.g., matching a target volatility target).
2. **Estimated** from a separate window, not the evaluation window.
3. **Reported as a sensitivity dimension** across a documented grid.

The chosen approach must be declared before evaluation (invariant `III.2`).

**Default:** `λ` calibrated to a 10% target annualized volatility on the training window. Grid sensitivity reported across `λ ∈ {2, 5, 10, 20}`.

---

## 5. Decision Rule Specification

### 5.1 Formal decision problem

Given the predictive distribution `p(r_t | F_t)` and the mean-variance utility, the decision problem is:

```
w_t* = argmax_{w ∈ W}   w^T μ_t − (λ / 2) · w^T Σ_t w   −   C(w, w_{t-1})
```

where:

- `μ_t = E[r_t | F_t]` is the predictive mean
- `Σ_t = Cov[r_t | F_t]` is the predictive covariance
- `C(w, w_{t-1})` is the transaction cost

### 5.2 Solution for the unconstrained case

Ignoring costs and constraints, the optimal weights are:

```
w_t* = (1 / λ) · Σ_t^{-1} · μ_t
```

This is the classical mean-variance portfolio. It is the reference solution.

### 5.3 Solution with constraints and costs

When constraints and costs are active, the problem is solved numerically as a convex program:

- **Objective:** quadratic in `w` (mean-variance term) plus a convex cost penalty.
- **Constraints:** linear (long-only, budget, position limits).
- **Turnover cap:** linear and `l1`-based; slack variables may be required.

**Solver requirement:** any solver used must be:
- Deterministic (same inputs → same output).
- Well-documented (name, version, tolerances).
- Reproducible under fixed random seeds (invariant `IV.2`).

**Default solver:** `cvxpy` with the default conic solver, or `scipy.optimize.minimize` with SLSQP. Choice documented in `08_...`.

### 5.4 Predictive distribution inputs

The decision rule consumes `μ_t` and `Σ_t`. These are produced by the model. The model is responsible for:

- Producing a calibrated predictive mean.
- Producing a calibrated predictive covariance.
- Representing uncertainty honestly. If the model cannot produce a full covariance, a documented shrinkage or factor structure is used.

For regime-aware models, the mixture distribution is:

```
μ_t = Σ_k π_t(k) · μ_k
Σ_t = Σ_k π_t(k) · ( Σ_k + (μ_k − μ_t)(μ_k − μ_t)^T )
```

This is the law of total expectation and total covariance for a mixture. It is not optional; ignoring the between-regime term understates uncertainty.

### 5.5 Forbidden allocation rules

The following rules are **prohibited** as primary allocation rules:

- "Long positive predictions with equal weight."
- "Long positive predictions with magnitude weight."
- "Long top-N predicted returns."
- Any rule that does not solve the objective in 3.2.

These rules may be reported as **diagnostic baselines** only, clearly labeled as non-optimal and not derived from the objective.

---

## 6. Constraint Specification

### 6.1 Admissible set `W`

The admissible set for v2.0:

```
W = { w ∈ R^N :  w_i ≥ 0              (long-only, no shorting)
             ,  Σ w_i ≤ 1             (fully-invested cap, cash allowed)
             ,  w_i ≤ w_max           (position limit, default w_max = 0.40)
             ,  ||w − w_{t-1}||_1 ≤ τ_max   (turnover cap, default τ_max = 0.50)
             ,  Σ w_i ≥ w_min          (minimum deployment, default w_min = 0.00)
             }
```

### 6.2 Constraint rationales

| Constraint | Rationale | Source |
|---|---|---|
| `w_i ≥ 0` | Long-only scope decision. | `03`, scope |
| `Σ w_i ≤ 1` | No leverage. Cash allowed. | `03`, scope |
| `w_i ≤ 0.40` | Prevents concentration in a single factor. | `04`, assumption D4/D5 |
| Turnover cap | Limits cost and enforces realistic trading. | `04`, invariant II.3 |
| Minimum deployment | Prevents degenerate all-cash solutions unless justified. | `04`, decision D6 |

### 6.3 Constraints are invariants

Every constraint in 6.1 is enforced at every rebalance. A solution that violates any constraint invalidates the run (invariant `II.2`).

### 6.4 No relaxed constraints

Constraints cannot be relaxed silently. If a constraint is found to bind frequently, this is reported as a diagnostic, not corrected by loosening the constraint after the fact.

### 6.5 Cash treatment

Cash is a residual asset with zero return (or risk-free return, if modeled). The constraint `Σ w_i ≤ 1` means cash is not an explicit decision variable; it absorbs the residual. The decision rule may explicitly output a cash weight `w_cash = 1 − Σ w_i` for reporting.

**Open question:** Should cash be an explicit asset in the decision, or a residual? See Section 12.

---

## 7. Cost Model Specification

### 7.1 Cost function

The transaction cost is:

```
C(w, w_{t-1}) = Σ_i  c_i · |w_i − w_{t-1,i}|
```

where `c_i` is the per-unit cost for asset `i`.

### 7.2 Cost parameter calibration

Two modes, both allowed, both reported distinctly:

**Mode 1: Assumed costs (default).**
- `c_i = c_assumed` for all `i`, with `c_assumed ∈ {5, 10, 20} bps`.
- Reported as "assumed cost scenario."
- Used as the baseline.

**Mode 2: Calibrated costs (robustness).**
- `c_i` calibrated from realized ETF spreads (e.g., average effective spread over the training window).
- Reported as "calibrated cost scenario."
- Used for robustness.

The mode used in the headline result must be declared before evaluation. If both modes are reported, the headline result is the assumed mode, with the calibrated mode as a robustness check.

### 7.3 What the cost model does not capture

- **Market impact.** Modeled as zero in the base case; addressed via capacity analysis.
- **Slippage.** Ignored in the base case; addressed via cost sensitivity.
- **Taxes.** Out of scope for this project.
- **Borrow costs.** Not applicable (long-only, no shorts).
- **Financing costs.** Not applicable (no leverage).
- **Regime-dependent costs.** Base case assumes constant; regime-conditional cost model is a robustness variant (`E5` in `04_...`).

### 7.4 Turnover reporting

Turnover is reported at every rebalance:

```
turnover_t = (1/2) · ||w_t − w_{t-1}||_1
```

Turnover is a first-class metric, not a secondary diagnostic. Invariant `II.4`.

### 7.5 Net vs gross returns

Headline performance metrics are always **net of costs**. Gross metrics may be reported as a diagnostic, clearly labeled, and never as the primary result. Invariant `II.3`.

---

## 8. Risk Penalty Specification

### 8.1 Risk penalty

Two options for the risk penalty `R(w)`:

**Option A — Variance:**
```
R(w) = w^T Σ_t w
```
Already part of the mean-variance utility. Under this option, `λ` in the utility and the risk penalty are the same parameter.

**Option B — CVaR:**
```
R(w) = CVaR_α( w^T r_t )
```
Under this option, the utility term `E[U(w^T r)]` becomes the mean, and CVaR is added as a separate penalty with its own parameter.

### 8.2 Default

**Default:** Option A (variance). Matches the mean-variance utility default.

**Robustness variant:** Option B (CVaR) as an alternative decision rule. Results compared in `07_...`.

### 8.3 Risk penalty is active

The risk penalty must be active. If `λ = 0`, this must be explicitly declared and justified. Invariant `II.5`.

---

## 9. Decision Rule Protocol

The following steps define the decision pipeline. Each step is a stage in the code, and each stage must be testable in isolation.

1. **Receive predictive distribution.** Model outputs `μ_t` and `Σ_t` (or the full mixture).
2. **Estimate expected utility.** Compute `E[U(w^T r_t)]` under the predictive distribution.
3. **Compute cost term.** Compute `C(w, w_{t-1})`.
4. **Compute risk penalty.** Compute `R(w)`.
5. **Solve constrained optimization.** Obtain `w_t*`.
6. **Verify constraints.** Assert `w_t* ∈ W`.
7. **Record decision.** Store `w_t*`, solver status, iteration count, and objective value.
8. **Compute realized return.** After observing `r_t`, compute `w_t*^T r_t − C(w_t*, w_{t-1})`.

Steps 1–7 happen before the decision is observable. Step 8 happens after. This temporal separation is enforced in code.

### 9.1 Determinism requirement

Given the same inputs (`F_t`, `w_{t-1}`, model parameters, seeds), the decision rule must produce the same output. Non-determinism in the solver must be eliminated or documented.

### 9.2 Numerical tolerance

All solver tolerances must be documented. If a solve fails to converge, this must be recorded and the failure handled according to a documented fallback rule.

### 9.3 Fallback rule

If the solver fails at time `t`:

1. Log the failure.
2. Fall back to `w_t = w_{t-1}` (no change). This is conservative and reflects the cost of a missed rebalance.
3. Mark the rebalance as failed in the results.

Any rebalance that used the fallback rule must be flagged in the final report. If more than a documented fraction of rebalances use the fallback, the run is invalid.

**Default fraction:** 5%.

---

## 10. Ties to Other Documents

| Downstream concern | Where specified |
|---|---|
| Which data features are allowed | `06_data_information_and_target_contract.md` |
| How the predictive distribution is produced | `08_minimum_viable_baseline_end_to_end_pipeline_architecture.md` |
| How the decision rule is evaluated | `07_evaluation_framework_and_backtesting_protocol.md` |
| What counts as success | `02`, success criteria; `09`, stop criteria |
| What counts as a valid run | `04`, invariants |

If a decision rule choice in this document conflicts with another document, this document is authoritative for the decision rule. Conflicts must be resolved by updating the other document.

---

## 11. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Objective is expected utility net of costs. | Matches primitive problem in `01`. |
| D2 | Default utility is mean-variance. | Standard, tractable, matches available information. |
| D3 | Default decision rule solves the constrained mean-variance problem. | Only rule derived from the objective. |
| D4 | Constraints are long-only, no leverage, turnover-capped, position-limited. | Matches scope in `03`. |
| D5 | Costs are `Σ_i c_i · |Δw_i|` with `c_i` assumed at 10 bps baseline. | Simple, transparent, well-documented. |
| D6 | Default risk penalty is variance; CVaR tested as robustness. | Matches utility default; CVaR addresses fat tails. |
| D7 | Ad hoc allocation rules are prohibited as primary. | Enforces invariant `II.1`. |
| D8 | Decision rule must be frozen before evaluation. | Enforces invariant `III.2`. |
| D9 | Solver must be deterministic and documented. | Reproducibility. |
| D10 | Solver failure fallback is `w_t = w_{t-1}`, capped at 5% of rebalances. | Conservative and auditable. |
| D11 | Gross returns never reported as headline. | Enforces invariant `II.3`. |
| D12 | Turnover is a first-class metric. | Enforces invariant `II.4`. |

---

## 12. Open Questions

1. Should cash be an explicit decision variable, or a residual of the long-only constraint? Explicit treatment would allow the optimizer to consider the risk-free rate directly but complicates the constraint formulation.
2. Is `λ` calibrated, estimated, or swept? Current default is calibrated to a target volatility on the training window, but this is not fully specified.
3. Should the cost model allow asset-specific and time-varying costs in the baseline, or only in robustness? Current default: constant per-asset.
4. Is the turnover cap `τ_max = 0.50` too tight or too loose? It was chosen as a placeholder and needs justification.
5. Should the decision rule optimize for CVaR by default, given `B5` (fat tails)? Current default is mean-variance with CVaR as a robustness check. This may be reversed after empirical testing.
6. What is the correct fallback behavior if the predictive covariance `Σ_t` is not positive semi-definite (a common numerical issue)? Current answer: shrink to nearest PSD matrix using a documented method, and log the event.
7. Does the decision rule need to account for regime uncertainty directly (i.e., condition the constraints on the regime), or is it sufficient to use the mixture mean and covariance? Current default: mixture mean/covariance.
8. Should solver failure handling distinguish between "infeasible" and "convergence failure"? Current answer: no, both trigger the fallback. May need refinement.
9. Is the 5% fallback threshold appropriate, or should it be tighter given the short sample?
10. Is the mean-variance formulation consistent with the fat-tail assumption `B5`, or does it require a distributional adjustment (e.g., using a Student-t covariance)? Current answer: the mixture covariance already carries some tail information; explicit tail adjustments are a robustness variant.

---

## 13. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial objective, utility, and decision specification for v2. |

---

## 14. References

- `00_docs_index.md` — documentation map and conventions.
- `01_first_principles_problem_decomposition.md` — primitive decomposition.
- `02_problem_framing.md` — formal problem statement.
- `03_scope_and_non_goals.md` — scope boundaries.
- `04_assumptions_and_invariants.md` — assumptions catalog and invariant set.
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2, 21–42.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Ang, A. (2014). *Asset Management: A Systematic Approach to Factor Investing*. Oxford University Press.
