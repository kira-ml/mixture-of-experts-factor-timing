# Scope and Non-Goals

**Project:** moe-factor-timing v2
**Status:** Draft
**Last Updated:** 2026-08-03

---

## 1. Purpose

This document defines the boundaries of v2.

Its job is to prevent three failure modes that plagued v1:

1. **Scope creep** — the project slowly expands to cover more assets, more frequencies, more models, more features, until nothing is done rigorously.
2. **Implicit claims** — the project implies things it never explicitly tests or supports.
3. **Undefined non-goals** — the absence of a boundary means any reviewer can project their own expectations onto the project and judge it by those expectations.

Scope defines what v2 **will** do.
Non-goals define what v2 **will not** do, and why.

Both are equally important. A project without non-goals is a project without scope.

---

## 2. Context and Dependencies

- **Depends on:** `00_docs_index.md`, `01_first_principles_problem_decomposition.md`, `02_problem_framing.md`
- **Feeds into:** `04_assumptions_and_invariants.md`, `05_objective_utility_and_decision_specification.md`
- **Reference only:** v1 `problem_framing.md`, v1 `README.md`, v1 `TODO.md`

Every inclusion and exclusion in this document must be traceable to a decision in `02_problem_framing.md` or a primitive in `01_first_principles_problem_decomposition.md`. If an item cannot be traced, it does not belong in scope.

---

## 3. Scope Overview

v2 is a **decision-focused, regime-aware benchmark for equity factor allocation under uncertainty**.

The project is a **methodological and empirical study**, not a product, not a fund, not a trading system.

The three words that define scope are:

- **Decision-focused** — the objective is allocation quality under uncertainty, not forecasting accuracy.
- **Regime-aware** — latent state is modeled explicitly, but regimes are treated as a hypothesis, not an axiom.
- **Benchmark** — the contribution is a reproducible framework, not a claimed strategy.

Anything outside these three words is out of scope unless explicitly included below.

---

## 4. In Scope

### 4.1 Asset universe

| Item | Inclusion | Rationale |
|------|-----------|-----------|
| **Domain** | US equity factors | Tractable, well-documented, liquid. |
| **Proxies** | Tradable ETFs only | Excludes non-investable indices. |
| **Initial set** | SPY, IWD, MTUM, QUAL, USMV | Market, Value, Momentum, Quality, Low Volatility. |
| **Benchmark asset** | Cash / T-bills (via BIL or equivalent) | Needed for risk-free rate and long-only accounting. |
| **Factor definitions** | Documented and fixed | No ad-hoc redefinition during the project. |
| **VIX** | **Excluded** from investable universe | Not directly tradable. May be used as a feature only. |

Rationale for excluding VIX from the investable set: `02_problem_framing.md` Section 8.2 prohibits non-tradable instruments in the allocation universe.

### 4.2 Decision problem

| Item | Inclusion | Rationale |
|------|-----------|-----------|
| **Frequency** | Monthly rebalancing | Matches macro data cadence, reduces turnover. |
| **Allocation direction** | Long-only (initial version) | Simplifies constraints; long-short is deferred. |
| **Leverage** | Not allowed | Keeps the decision problem bounded and realistic. |
| **Shorting** | Not allowed in v2.0 | Deferred to a later version. |
| **Cash allocation** | Allowed | Long-only without cash is over-constrained. |
| **Turnover cap** | Modeled and enforced | Required for realistic cost treatment. |
| **Position limits** | Configurable, documented | Prevents degenerate solutions. |
| **Utility function** | Specified in `05_...` | Not fixed in this document. |
| **Cost model** | Included, calibrated or assumed | Reported alongside results. |

### 4.3 Information set

| Item | Inclusion | Rationale |
|------|-----------|-----------|
| **Historical returns** | Lagged, using only data ≤ `t-1` | No look-ahead. |
| **Realized volatility** | Computed from lagged returns only | No look-ahead. |
| **Macro indicators** | Vintage-aware (ALFRED or equivalent) | Publication lags and revisions modeled. |
| **Sentiment / liquidity** | Only if tradable and lagged | Avoid synthetic proxies without economic grounding. |
| **Current portfolio** | Always included | Required for turnover and cost modeling. |
| **Future information** | Excluded | Fundamental rule. |

### 4.4 Models

| Item | Inclusion | Rationale |
|------|-----------|-----------|
| **Naive baselines** | 1/N, persistence, rolling average, momentum | Minimum viable comparison set. |
| **Linear baselines** | Ridge, Lasso | Reproducible, interpretable. |
| **Tree baselines** | Random forest, gradient boosting | Non-linear but still reproducible. |
| **Regime-aware models** | HMM, GMM, Mixture of Experts | Core hypothesis under test. |
| **Decision-focused models** | MoE trained with decision-aware loss | Primary methodological contribution. |
| **Bayesian variants** | Optional, if time permits | Useful for uncertainty decomposition. |
| **Deep models** | Optional, isolated, not default | Kept behind a flag to preserve reproducibility. |

All models must expose the same interface and be evaluated under the same protocol.

### 4.5 Evaluation

| Item | Inclusion | Rationale |
|------|-----------|-----------|
| **Predictive metrics** | NLL, CRPS, calibration | Predictions are an input to decisions. |
| **Decision metrics** | Utility, Sharpe, Sortino, MaxDD, Calmar | Decision quality is the objective. |
| **Cost-aware metrics** | Turnover-adjusted returns | Costs materially change rankings. |
| **Capacity metrics** | Capacity-adjusted return | Prevents illusion of infinite scale. |
| **Statistical tests** | DM, SPA, deflated Sharpe, bootstrap CIs | Required for honest claims. |
| **Robustness** | Multi-window, multi-seed, multi-hyperparameter | Required for robustness claims. |
| **Regime analysis** | Stability, interpretability, persistence | Required for regime claims. |

### 4.6 Deliverables

| Item | Inclusion |
|------|-----------|
| Pre-code documentation | `docs/00` through `docs/09` |
| Data pipeline | Vintage-aware, lag-correct |
| Model implementations | All models in Section 4.4 |
| Backtest engine | Walk-forward, cost-aware, constraint-aware |
| Evaluation module | Predictive, decision, statistical |
| Regime analysis module | Interpretability and stability |
| Reproducible runs | Configs + environment capture |
| Public report | Bounded claims, honest limitations |
| Open-source release | MIT licensed |

---

## 5. Out of Scope

### 5.1 Asset classes and instruments

| Item | Status | Reason |
|------|--------|--------|
| International equity factors | Out | Different data, different microstructure, different regimes. |
| Fixed income | Out | Different risk structure, not comparable. |
| Commodities | Out | Different economics. |
| FX | Out | Different data availability, different regime dynamics. |
| Crypto | Out | Short history, different microstructure, different risk. |
| Options | Out | Adds a large decision-space dimension. |
| Futures | Out (except for cash proxy) | Roll costs and margin complicate the problem. |
| Single stocks | Out | Factor-level focus, not security selection. |
| VIX spot | Out | Not tradable. |
| Non-US markets | Out | Different regime structure. |

### 5.2 Frequencies

| Item | Status | Reason |
|------|--------|--------|
| Daily rebalancing | Out | Costs dominate; macro signal is not daily. |
| Weekly rebalancing | Out | Insufficient signal-to-noise at macro level. |
| Intraday | Out | Different problem entirely. |
| Quarterly | Deferred | Monthly is the initial frequency. |
| Annual | Out | Too coarse. |

### 5.3 Decision mechanisms

| Item | Status | Reason |
|------|--------|--------|
| Short selling | Out in v2.0 | Adds constraint complexity; deferred. |
| Leverage | Out | Changes the risk profile; deferred. |
| Dynamic risk targeting | Deferred | Requires robust volatility forecast; later version. |
| Drawdown control | Deferred | Requires dynamic optimization; later version. |
| Multi-period optimization | Deferred | Single-period is the initial formulation. |
| Reinforcement learning | Out | Different problem framing; not needed initially. |

### 5.4 Claims

| Item | Status | Reason |
|------|--------|--------|
| "Beats the market" | Out | Not a valid claim from a backtest. |
| "Alpha exists" | Out | Requires a specific benchmark and significance. |
| "MoE is superior" | Out | One model class among several. |
| "Regimes are real" | Out | Regimes are a hypothesis, not a fact. |
| "Live performance" | Out | Backtest ≠ live. |
| "Scalable to any AUM" | Out | Capacity is a constraint, not a claim. |
| "Statistically significant outperformance" | Out unless tests support it | Requires power analysis. |

### 5.5 Engineering and infrastructure

| Item | Status | Reason |
|------|--------|--------|
| Production deployment | Out | Not a product. |
| Real-time trading | Out | Not a fund. |
| Broker integration | Out | Not connected to markets. |
| Live dashboards | Out | Not a monitoring system. |
| Model serving APIs | Out | Not a service. |
| Cloud infrastructure | Out | Local reproducibility is sufficient. |
| Docker / Kubernetes | Deferred | Only if needed for reproducibility. |
| MLflow / experiment servers | Deferred | Only if manual tracking becomes insufficient. |

---

## 6. Deferred Items

These items are neither in scope nor out of scope. They are explicitly deferred. Deferral means: **not in v2.0, possibly in a later version, and only if the current scope is completed rigorously.**

| # | Deferred Item | Trigger Condition for Reconsideration |
|---|---------------|----------------------------------------|
| F1 | Long-short allocation | Long-only version is complete and validated. |
| F2 | Leverage | Utility-based risk targeting is implemented. |
| F3 | Multi-period optimization | Single-period is stable. |
| F4 | Dynamic risk targeting | Volatility forecasts are calibrated. |
| F5 | International factors | US version is complete and reproducible. |
| F6 | Deep learning models | Simpler models are exhausted. |
| F7 | Reinforcement learning | Decision problem is well-characterized. |
| F8 | Options overlays | Not planned; would change the problem. |
| F9 | Live trading | Out of scope for the project's lifetime. |
| F10 | Docker / cloud deployment | Only if reproducible locally fails. |

Deferral is not a promise. It is a note that the decision to include or exclude has not yet been made.

---

## 7. Boundary Conditions

Some items look like they belong inside the scope but require special handling.

### 7.1 VIX as a feature

- **In scope:** VIX may be used as a feature.
- **Out of scope:** VIX as an investable asset.
- **Rule:** Any feature derived from VIX must be lagged to reflect actual availability.

### 7.2 Macro data revisions

- **In scope:** Vintage-aware macro data.
- **Out of scope:** Revised macro data used as if it were available in real time.
- **Rule:** If vintage data is unavailable for a series, the series is either dropped or lagged conservatively.

### 7.3 Cost model calibration

- **In scope:** Modeling and reporting costs.
- **Out of scope:** Claims that the cost model reflects any specific broker or venue.
- **Rule:** Costs are labeled as either "assumed" or "calibrated," and the distinction is reported.

### 7.4 Backtest performance

- **In scope:** Reporting backtest results.
- **Out of scope:** Treating backtest results as evidence of live performance.
- **Rule:** Every reported backtest metric is accompanied by an explicit disclaimer of its limitations.

### 7.5 Number of regimes

- **In scope:** Testing multiple values of `K`.
- **Out of scope:** Claiming that any specific `K` is the "true" number of regimes.
- **Rule:** Model selection uses out-of-sample criteria, not backtest Sharpe.

### 7.6 Model complexity

- **In scope:** Comparing models of varying complexity.
- **Out of scope:** Adding complexity without evidence it improves decisions.
- **Rule:** Every added model must be justified by a specific hypothesis it tests.

---

## 8. Scope Discipline

### 8.1 The scope test

Before adding any item to the project, it must pass this test:

1. Can it be traced to a primitive in `01_first_principles_problem_decomposition.md`?
2. Does it answer a research question in `02_problem_framing.md`?
3. Is it compatible with the information set defined in `02_problem_framing.md`?
4. Can it be evaluated under the protocol in `07_evaluation_framework_and_backtesting_protocol.md`?
5. Would its exclusion reduce the validity of the project's claims?

If any answer is no, the item is out of scope or deferred.

### 8.2 The non-goal test

Before disclaiming any item, it must pass this test:

1. Is the disclaimer necessary to prevent a plausible misreading?
2. Is it specific enough to be falsifiable?
3. Does it correspond to a real capability the project could plausibly be expected to have?

Non-goals that fail this test are noise. Non-goals that pass this test are essential.

### 8.3 Scope change protocol

Scope changes require:

1. A written justification in the decision log.
2. An update to this document.
3. A corresponding update to `02_problem_framing.md` if the change affects the research questions.
4. A corresponding update to `07_evaluation_framework_and_backtesting_protocol.md` if the change affects evaluation.

Scope cannot change silently. Silent scope change is the mechanism by which projects overclaim.

---

## 9. What This Project Is Not

Explicit, falsifiable non-goals for public communication:

1. This project is **not** a trading strategy.
2. This project is **not** a fund.
3. This project is **not** investment advice.
4. This project does **not** claim to beat any benchmark.
5. This project does **not** claim that regimes exist in the market.
6. This project does **not** claim that MoE is superior to any other model class.
7. This project does **not** claim that backtest results imply live performance.
8. This project does **not** claim that the cost model reflects actual trading costs.
9. This project does **not** claim that the latent states are economically interpretable without evidence.
10. This project does **not** claim statistical significance without tests.

These statements should appear verbatim in the final report.

---

## 10. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Asset universe is US equity factors via tradable ETFs. | Tractable, liquid, well-documented. |
| D2 | VIX is excluded from the investable universe, allowed as a feature. | VIX spot is not tradable. |
| D3 | Monthly rebalancing only. | Matches macro cadence; reduces cost dominance. |
| D4 | Long-only with cash allowed. | Simplifies initial constraints; shorting deferred. |
| D5 | No leverage. | Keeps the decision problem bounded. |
| D6 | Costs are always modeled and reported. | Zero-cost backtests are invalid. |
| D7 | Vintage-aware macro data is required. | Otherwise the backtest is invalid. |
| D8 | Deferred items are tracked explicitly. | Prevents silent scope expansion. |
| D9 | Scope changes require written justification. | Prevents silent drift. |
| D10 | Public non-goals are stated verbatim in the report. | Prevents plausible misreadings. |

---

## 11. Open Questions

1. Should cash be treated as an asset in the allocation decision, or only as a residual?
2. Should the initial version include a simple long-short formulation as a sanity check, even if not the primary mode?
3. Is "monthly" the correct frequency, or should the project also test quarterly for robustness?
4. Should factor definitions be based on the ETF sponsors' indices, or on the academic factor definitions?
5. Should the project include a "do nothing" strategy (100% cash) as a baseline?
6. What is the correct response if vintage data is unavailable for a required series — drop the series or lag it?
7. Should the project define a fixed cost model, or compare against multiple cost scenarios?
8. Is "no leverage" a scope decision or a constraint decision that belongs in `05`?

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial scope and non-goals for v2. |

---

## 13. References

- `00_docs_index.md` — documentation map and conventions.
- `01_first_principles_problem_decomposition.md` — primitive decomposition.
- `02_problem_framing.md` — formal problem statement and research questions.
- v1 `problem_framing.md` — reference only.
- v1 `README.md` — reference only.
