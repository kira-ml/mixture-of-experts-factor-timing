# Problem Framing

**Project:** moe-factor-timing v2
**Status:** Draft
**Last Updated:** 2026-08-03

---

## 1. Purpose

This document converts the first-principles decomposition in `01_first_principles_problem_decomposition.md` into a formal problem statement, a set of research questions, and a specification of what will be built, measured, and claimed.

Its job is to remove ambiguity about:

- What problem is being solved.
- Why it is worth solving.
- What is inside and outside the scope.
- What form a solution must take.
- What evidence would count as progress.

This document does **not** choose models or architectures. Those are downstream decisions. It defines the problem those choices must serve.

---

## 2. Context and Dependencies

- **Depends on:** `00_docs_index.md`, `01_first_principles_problem_decomposition.md`
- **Feeds into:** `03_scope_and_non_goals.md`, `04_assumptions_and_invariants.md`, `05_objective_utility_and_decision_specification.md`
- **Reference only:** v1 `problem_framing.md`, v1 `README.md`, v1 `TODO.md`

Where v1 and v2 differ, v2 supersedes. Where v1 made an unexamined choice, v2 either justifies it or replaces it.

---

## 3. Relationship to v1

v1 asked:

> Can a probabilistic model that explicitly represents uncertainty over economic regimes provide a coherent framework for factor timing compared to simpler deterministic approaches?

That question has three weaknesses:

1. It presupposes that "factor timing" is the problem. The primitive problem is allocation, not factor timing.
2. It presupposes that probabilistic regime models are the candidate solution. Probabilistic models are one class of tools among several.
3. It measures success by comparison to baselines rather than by decision quality under uncertainty.

v2 asks a sharper question:

> Given finite capital, an admissible set of allocations, non-stationary and partially observable markets, and non-zero costs of action, does explicitly modeling latent state and model uncertainty improve out-of-sample allocation decisions, measured by risk-adjusted utility after costs and validated with statistical discipline?

This is the question v2 is built to answer. Everything downstream — data, models, backtests, metrics — serves this question.

---

## 4. The Problem

### 4.1 Plain-language statement

An investor must allocate capital across a set of tradable assets at regular intervals. The future is uncertain, the relationship between information and outcomes changes over time, and changing the allocation costs money. The investor wants a decision rule that produces good outcomes over time, using only information that was actually available at the time of each decision, and that can be judged honestly after the fact.

That is the problem.

### 4.2 Why this problem is hard

Four distinct difficulties compound:

1. **Return uncertainty.** Even with a perfect model of the world, future returns are noisy.
2. **Latent state.** The current economic, liquidity, and sentiment conditions are not directly observable.
3. **Non-stationarity.** The mapping from information to outcomes changes over time.
4. **Cost of action.** Every reallocation consumes wealth, so a noisy signal can be worse than no signal.

Each difficulty is primitive. None can be assumed away without changing the problem.

### 4.3 Why the problem matters

The problem is fundamental to systematic investing. Even modest improvements in decision quality, sustained over time, compound into meaningful economic value. But the same compounding works in reverse: a decision rule that looks good in a backtest but fails out-of-sample is worse than no rule at all.

The value of this project is not a claimed outperformance. The value is a rigorous, reproducible framework for evaluating whether a class of models — probabilistic, regime-aware, uncertainty-explicit — is decision-useful under realistic constraints.

---

## 5. Formal Specification

### 5.1 Time and information

Let `t = 0, 1, 2, ...` index rebalancing periods (initially monthly).

Let `F_t` denote the information set available at the start of period `t`. It contains only information that was **actually observable** at that time:

- Historical returns of all assets up to and including `t-1`
- Realized volatility and correlation estimates up to `t-1`
- Macroeconomic indicators as they were published (vintage-aware)
- Current portfolio weights `w_{t-1}`

`F_t` excludes:

- Future returns
- Revised macroeconomic data not yet published at `t`
- Any same-period information that a live investor would not have had

### 5.2 Assets and allocations

Let `A = {a_1, ..., a_N}` be the set of investable assets.

Let `r_t ∈ R^N` be the vector of realized returns over period `t`.

Let `W ⊆ R^N` be the admissible set of portfolio weights. Initially:

- Long-only: `w_i ≥ 0`
- Fully invested: `Σ w_i = 1`
- Turnover cap: `||w_t − w_{t-1}||_1 ≤ τ_max`

The admissible set is a design choice and is documented in `05_objective_utility_and_decision_specification.md`.

### 5.3 Latent state

Let `S_t` be a latent state variable representing the unobserved condition of the market at time `t`. Its form — discrete, continuous, hybrid — is a modeling choice, not a primitive.

Let `π_t(k) = P(S_t = k | F_t)` denote the posterior over latent states given available information.

### 5.4 Return model

The joint distribution of returns is assumed to depend on the latent state:

```
r_t | S_t = k, F_t  ~  D_k( μ_k(F_t), Σ_k(F_t) )
```

where `D_k` is a distributional family (Gaussian, Student-t, or other), and `μ_k`, `Σ_k` are functions of the information set.

The predictive distribution is the mixture over latent states:

```
p(r_t | F_t) = Σ_k π_t(k) · D_k( μ_k(F_t), Σ_k(F_t) )
```

The form of `π_t`, `μ_k`, `Σ_k`, and `D_k` is not fixed by this framing. It is a modeling choice to be justified downstream.

### 5.5 Decision problem

Given the predictive distribution and the current portfolio, the investor chooses weights:

```
w_t* = argmax_{w ∈ W}  E[ U(w^T r_t) | F_t ]  −  C(w, w_{t-1})  −  λ · R(w)
```

where:

- `U(·)` is the investor's utility function
- `C(·,·)` is the transaction and impact cost model
- `R(·)` is a risk penalty
- `λ ≥ 0` is a risk aversion parameter

The specific forms of `U`, `C`, `R`, and `λ` are documented in `05_objective_utility_and_decision_specification.md`.

### 5.6 Evaluation

The decision process is evaluated on an out-of-sample period using:

- **Predictive metrics** — calibration, likelihood, CRPS. RMSE and MAE are reported as secondary diagnostics, not as objectives.
- **Decision metrics** — out-of-sample utility, Sharpe, Sortino, maximum drawdown, Calmar, turnover, capacity-adjusted return.
- **Statistical metrics** — confidence intervals, multiple-testing adjustments, and tests for whether observed differences exceed what chance would produce.

Evaluation protocol is documented in `07_evaluation_framework_and_backtesting_protocol.md`.

---

## 6. The Four Sub-Problems

The formal problem decomposes into four sub-problems. Each is separately testable. Each can fail independently.

### 6.1 Representation

**Question:** How is the state of the world represented in a form the model can use?

**Concerns:**
- Choice of features and their transformations
- Whether the latent state is discrete, continuous, or hybrid
- How non-stationarity is encoded
- Whether the representation is stable across time

**Failure modes:**
- Features contain look-ahead information
- Features are unstable across regimes
- Representation is too rich and overfits
- Representation is too coarse and misses signal

### 6.2 Inference

**Question:** How is the latent state and its uncertainty estimated from observations?

**Concerns:**
- Model class (parametric, semi-parametric, non-parametric)
- Parameter estimation method
- Representation of parameter uncertainty
- Representation of model uncertainty

**Failure modes:**
- Overfitting to in-sample noise
- Underfitting due to excessive regularization
- Failure to represent uncertainty honestly
- Model selection driven by the backtest it is meant to be validated on

### 6.3 Decision

**Question:** How is the inferred distribution mapped to an allocation?

**Concerns:**
- Choice of utility function
- Choice of risk penalty
- Cost model
- Constraint set

**Failure modes:**
- Decision rule not derived from an objective
- Utility function inconsistent with risk tolerance
- Costs underestimated
- Constraints assumed but not enforced

### 6.4 Evaluation

**Question:** How is the decision process judged honestly?

**Concerns:**
- Out-of-sample protocol
- Metric selection
- Statistical significance
- Robustness across windows and hyperparameters
- Multiple testing

**Failure modes:**
- No multiple-testing correction
- Short out-of-sample period
- Metrics chosen after seeing results
- No robustness checks
- Claims of significance without tests

v1 collapsed all four sub-problems into a single pipeline. v2 keeps them separate so that each can be diagnosed independently.

---

## 7. Research Questions

### 7.1 Primary

> Does explicitly modeling latent state and model uncertainty improve out-of-sample allocation decisions, measured by risk-adjusted utility after costs and validated statistically, compared to deterministic, point-forecast, or heuristic alternatives?

### 7.2 Secondary

1. **Decision vs prediction.** Does optimizing for decision utility produce better allocations than optimizing for predictive accuracy?
2. **Uncertainty decomposition.** Which source of uncertainty — regime, parameter, model, or return noise — contributes most to allocation error?
3. **Feature value.** Do macroeconomic indicators add incremental decision value after accounting for publication lags and revisions?
4. **Regime interpretability.** Are learned latent states economically interpretable and stable across windows and specifications?
5. **Robustness.** How sensitive is performance to training window, latent state count, cost model, and allocation rule?
6. **Statistical honesty.** Can any observed performance improvement be distinguished from chance after multiple-testing correction?

### 7.3 Non-questions

The following are explicitly **not** research questions for v2:

- Can MoE beat the market?
- Does a backtest Sharpe above 1 imply skill?
- Is K=4 the optimal number of regimes?
- Can the model predict VIX?
- Is the v1 configuration optimal?

These are either unanswerable with available data or are artifacts of the v1 framing.

---

## 8. What v2 Will and Will Not Do

### 8.1 v2 will

- Frame the problem as decision-focused allocation under uncertainty.
- Keep representation, inference, decision, and evaluation separate.
- Use vintage-aware data to avoid look-ahead bias.
- Pre-commit to evaluation protocol before running backtests.
- Report performance with confidence intervals and multiple-testing adjustments.
- Compare model classes under a common decision rule.
- Publish code, data, and configuration as an open-source benchmark.

### 8.2 v2 will not

- Claim to have discovered a market-beating strategy.
- Use the backtest as both selection criterion and validation set.
- Include non-tradable instruments (e.g., VIX spot) in the investable universe.
- Ignore transaction costs, turnover, or capacity.
- Treat 42 out-of-sample months as statistical evidence.
- Present backtest metrics as live performance.
- Assume that regime models are inherently better than alternatives.

---

## 9. Success Criteria

These are criteria for whether the **project** succeeds, not for whether any particular model wins. Model-level criteria are documented in `07_evaluation_framework_and_backtesting_protocol.md`.

| # | Criterion | Definition |
|---|-----------|------------|
| S1 | No look-ahead bias | All features and targets respect the information set `F_t` at each decision. |
| S2 | Pre-committed evaluation | Backtest protocol is frozen before any model is evaluated. |
| S3 | Decision-focused metrics | Performance is measured by utility and risk-adjusted return, not RMSE alone. |
| S4 | Realistic costs | Turnover, spread, and impact are modeled and reported. |
| S5 | Statistical honesty | Performance is reported with uncertainty and multiple-testing adjustment. |
| S6 | Robustness | Results hold across reasonable windows, hyperparameters, and subperiods. |
| S7 | Interpretability | Latent states are analyzed, not treated as black boxes. |
| S8 | Reproducibility | Code, data, and configuration reproduce reported results. |
| S9 | No overclaiming | Claims match evidence and are explicitly bounded. |

A project that satisfies S1–S9 but produces no performance improvement is a **successful negative result**. A project that produces strong performance but violates S1–S9 is a failure.

---

## 10. Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| D1 | Problem framing | This document, plus `01`, `03`, `04`, `05`. |
| D2 | Data pipeline | Vintage-aware, lag-correct, reproducible. |
| D3 | Model suite | Baselines, regime-aware models, decision-focused models. |
| D4 | Backtest engine | Walk-forward, cost-aware, constraint-aware. |
| D5 | Evaluation module | Predictive, decision, and statistical metrics. |
| D6 | Regime analysis | Stability, interpretability, economic characteristics. |
| D7 | Report | Honest, bounded, methodologically transparent. |
| D8 | Open-source release | MIT-licensed repository with reproducible results. |

---

## 11. Scope and Constraints (Preview)

Full scope is in `03_scope_and_non_goals.md`. Summary:

**In scope:**
- US equity factors and tradable ETFs
- Monthly rebalancing
- Long-only allocation (initial version)
- Macroeconomic indicators with publication lags
- Probabilistic regime-aware and decision-focused models
- Statistical robustness checks

**Out of scope:**
- Intraday or high-frequency trading
- Options and complex derivatives
- International markets (initial version)
- Live trading
- Claims of alpha generation

**Constraints:**
- Limited historical sample
- Non-stationarity
- Data revisions
- Transaction costs and capacity
- Model risk
- Multiple testing

---

## 12. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Frame the problem as decision-focused allocation, not return forecasting. | Primitives are capital, allocation, uncertainty, cost, non-stationarity. |
| D2 | Keep representation, inference, decision, and evaluation separate. | Coupling made v1 undiagnosable. |
| D3 | Treat regimes as a hypothesis, not an axiom. | Non-stationarity is primitive; regimes are one model of it. |
| D4 | Pre-commit to evaluation before running backtests. | Prevents selection bias. |
| D5 | Require vintage-aware data for macro features. | Eliminates look-ahead bias. |
| D6 | Report performance with statistical uncertainty. | 42 months is not evidence. |
| D7 | Treat successful negative results as project successes. | The benchmark's value is its rigor, not its returns. |
| D8 | Keep v1 as reference only. | v1's framing is superseded, not extended. |

---

## 13. Open Questions

1. Is the primary research question answerable at all with available data, or is it inherently underpowered?
2. Should the initial version restrict to long-only, or is a simple long-short formulation more informative?
3. Should the utility function be specified in this document or deferred to `05`?
4. Is "decision-focused" a strong enough framing, or should the project also commit to a specific decision rule family (e.g., mean-variance, CVaR)?
5. How should model uncertainty be represented without expanding the model class indefinitely?
6. What is the minimum out-of-sample length needed for the statistical tests to have power? Is it achievable?
7. Should the project pre-register its hypotheses in a separate document, or is the pre-code docs set sufficient?

---

## 14. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial v2 problem framing. Supersedes v1 `problem_framing.md`. |

---

## 15. References

- `00_docs_index.md` — documentation map and conventions.
- `01_first_principles_problem_decomposition.md` — primitive decomposition and v1 assumption audit.
- v1 `problem_framing.md` — reference only.
- Fama & French (1993) — factor model origins.
- Asness, Moskowitz & Pedersen (2013) — value and momentum everywhere.
- Ang & Bekaert (2002) — regime-switching in asset allocation.
- Jacobs, Jordan, Nowlan & Hinton (1991) — Mixture of Experts.
- Shih (2020) — *Machine Learning for Factor Investing*, CFA Institute Research Foundation.
