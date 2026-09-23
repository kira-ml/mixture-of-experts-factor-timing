# First-Principles Problem Decomposition

**Project:** moe-factor-timing v2
**Status:** Draft
**Last Updated:** 2026-08-03

---

## 1. Purpose

This document decomposes the factor timing problem from first principles.

It does **not** define the solution. It does **not** choose models. It does **not** propose an architecture. Its only job is to break the problem down to its most primitive components, separate what is fundamental from what is inherited convention, and identify the actual problem that needs to be solved.

The output of this document is the input to `02_problem_framing.md`.

The central discipline of this document is the question:

> **Is this a primitive, or is this a choice inherited from v1 that has never been justified?**

Anything that is a choice is marked as such and must be justified in a later document or discarded.

---

## 2. Context and Dependencies

- **Depends on:** `00_docs_index.md`
- **Feeds into:** `02_problem_framing.md`
- **Reference only:** v1 `README.md`, v1 `problem_framing.md`, v1 `TODO.md`

This document does not assume v1 was wrong. It assumes v1 made choices that were reasonable under uncertainty and are now being re-examined with the benefit of hindsight.

---

## 3. Method

First-principles decomposition proceeds in five steps.

1. **State the objective in plain language** without using domain jargon.
2. **Identify primitives** — things that cannot be reduced further without losing meaning.
3. **Separate primitives from derived concepts** — derived concepts are useful but not fundamental.
4. **Audit inherited assumptions** — list every assumption v1 made, and mark it as primitive, derived, or unexamined choice.
5. **Restate the root problem** in terms of primitives only.

Every claim in this document must survive the question: *"Is this true by definition, or is this a modeling choice?"*

---

## 4. Step 1 — The Objective in Plain Language

Strip away all domain terminology. What is an investor actually trying to do?

> An investor has capital. They can allocate it across a set of assets. They do not know which assets will do well. They must decide how much to put into each asset, knowing they might be wrong, knowing that changing their mind costs money, and knowing that the future may not look like the past.

That is the entire problem. Everything else is elaboration.

Notice what is present in that plain statement:

- Capital
- A set of assets
- Uncertainty about the future
- A decision about allocation
- A cost of changing the decision
- Non-stationarity

Notice what is **absent**:

- "Returns"
- "Factors"
- "Regimes"
- "Forecasts"
- "Models"
- "Sharpe ratios"

Those are all derived concepts. They are tools. They are not the problem.

---

## 5. Step 2 — Primitives

A primitive is something that cannot be removed without removing the problem itself.

### 5.1 Capital

There is a finite quantity of wealth to allocate. Without capital, there is no allocation problem.

**Primitive.** Cannot be removed.

### 5.2 A Set of Possible Allocations

There is a set of admissible ways to distribute capital. The set may be constrained, but it must exist.

**Primitive.** Cannot be removed.

### 5.3 Time

Allocation happens sequentially. Decisions at time `t` affect the wealth available at time `t+1`.

**Primitive.** Cannot be removed. A single-period problem is a degenerate special case.

### 5.4 Uncertainty About Future Outcomes

The outcome of any allocation is not known in advance. This is the entire reason the problem is hard.

**Primitive.** Cannot be removed. If the future were known, there is no problem — only an optimization.

### 5.5 Cost of Action

Changing an allocation costs something. This includes commissions, spreads, market impact, taxes, and opportunity cost.

**Primitive.** Cannot be removed in any realistic setting. A model that assumes zero cost is solving a different problem.

### 5.6 Non-Stationarity

The relationship between information and future outcomes changes over time. This is an empirical observation, not an assumption.

**Primitive** in the sense that it cannot be assumed away. Whether it can be modeled is a separate question.

### 5.7 Partial Observability

At any time `t`, the investor does not observe everything that determines future outcomes. Some state variables are hidden.

**Primitive.** Cannot be removed. If everything were observable, uncertainty would collapse to noise.

---

## 6. Step 3 — Derived Concepts

These are useful, but they are **not** primitives. Each one is a modeling choice that must be justified.

### 6.1 Returns

A return is a derived quantity: the change in wealth from holding an asset over a period.

**Derived.** Useful, but not fundamental. The primitive is wealth. Returns are a normalization.

### 6.2 Factors

A factor is a derived construct: a common source of variation in returns across assets.

**Derived.** Factors are a compression of the covariance structure of returns. They are not observed. They are inferred.

### 6.3 Regimes

A regime is a derived construct: a hypothesized latent state that governs the joint distribution of returns.

**Derived.** Regimes are one way to model non-stationarity. They are not the only way. They are not observed.

### 6.4 Forecasts

A forecast is a derived construct: a statement about the distribution of future outcomes given current information.

**Derived.** Forecasts are useful only insofar as they improve decisions.

### 6.5 Models

A model is a derived construct: a parameterized mapping from information to forecasts.

**Derived.** Models are tools. They are never the objective.

### 6.6 Sharpe Ratio

A Sharpe ratio is a derived construct: a specific function of the first two moments of a return series.

**Derived.** It is one summary of performance. It is not the objective. It ignores tails, skewness, drawdown, and capacity.

### 6.7 Alpha

Alpha is a derived construct: the portion of return not explained by a benchmark model.

**Derived.** Alpha is defined relative to a choice of benchmark. Change the benchmark, change the alpha.

The implication of this section is direct:

> **v1 was organized around derived concepts (factors, regimes, forecasts, MoE, Sharpe). The v2 decomposition should be organized around primitives (capital, allocation, time, uncertainty, cost, non-stationarity, partial observability).**

---

## 7. Step 4 — Assumption Audit of v1

Every assumption v1 made, classified by whether it was primitive, derived, or unexamined.

| # | v1 Assumption | Classification | Comment |
|---|---------------|----------------|---------|
| A1 | The problem is factor timing. | Derived | "Factor" is a modeling choice. The primitive is allocation across assets. |
| A2 | The prediction target is next-month returns. | Derived | Horizon and target are choices, not primitives. |
| A3 | Regimes are discrete and latent. | Derived | Continuous, unstable, or non-Markov states are equally plausible. |
| A4 | Regimes are Markov. | Unexamined | Transition structure was never tested against alternatives. |
| A5 | Four regimes are optimal. | Unexamined | K=4 was selected by backtest Sharpe, not by out-of-sample model selection. |
| A6 | VIX is a factor. | Unexamined | VIX spot is not directly investable. |
| A7 | Six factors suffice. | Derived | Factor count is a choice. |
| A8 | Monthly frequency is appropriate. | Derived | Horizon is a choice. |
| A9 | ETFs are valid factor proxies. | Derived | ETFs have tracking error, fees, and overlapping exposures. |
| A10 | Macro data are available at month-end. | Unexamined | Publication lags and revisions were not modeled. |
| A11 | 96 months of training is sufficient. | Unexamined | Chosen by backtest stability, not by learning theory or power analysis. |
| A12 | 42 out-of-sample months are sufficient. | Unexamined | Far too short for statistical inference. |
| A13 | 10 bps transaction costs are realistic. | Unexamined | Costs were not calibrated to actual instruments. |
| A14 | Magnitude-weighted long-only allocation is appropriate. | Unexamined | The allocation rule was not derived from an objective. |
| A15 | Sharpe ratio is the primary metric. | Unexamined | Ignores tails, drawdown, capacity, and skewness. |
| A16 | MoE with linear experts is the right model class. | Derived | One of many possible model classes. |
| A17 | Soft regime assignment is better than hard. | Derived | Soft assignment is a choice. It may or may not help decisions. |
| A18 | EM with 100 iterations converges. | Unexamined | Convergence was not formally checked. |
| A19 | The model should minimize RMSE. | Unexamined | RMSE is not the decision objective. |
| A20 | Backtest performance implies live performance. | Unexamined | This is the most dangerous assumption in the entire project. |

### 7.1 What the audit reveals

Three patterns:

1. **Derived concepts were treated as primitives.** Factors, regimes, and forecasts were treated as the problem, not as tools for solving the problem.
2. **Choices were made by backtest performance.** K=4, 96 months, 100 iterations, magnitude weighting — all selected by looking at the same backtest used to evaluate the model. This is selection bias.
3. **Statistical validity was not established.** No multiple-testing correction, no confidence intervals, no out-of-sample model selection protocol.

These are not failures of v1. They are the natural consequence of not writing this document first.

---

## 8. Step 5 — The Root Problem

Restated in primitives only:

> **At each point in time, an investor with finite capital must choose an allocation across a set of assets. The future is uncertain, partially unobservable, and non-stationary. Changing the allocation costs money. The investor must make this decision repeatedly, using only information that was actually available at the time, and must be able to judge afterward whether the decision process was sound.**

That is the problem.

Everything else — factors, regimes, MoE, Sharpe ratios — is a candidate tool for solving it.

### 8.1 What this reframing changes

| Old framing | New framing |
|---|---|
| Predict next-month factor returns. | Choose allocations that maximize expected utility under uncertainty. |
| Model regimes. | Model non-stationarity, of which regimes are one hypothesis. |
| Minimize RMSE. | Maximize out-of-sample risk-adjusted utility after costs. |
| Use MoE. | Compare model classes, including MoE, under a common decision rule. |
| Report Sharpe ratio. | Report utility, drawdown, turnover, capacity, and statistical significance. |
| Backtest with v1 configuration. | Pre-commit to a protocol before seeing results. |

### 8.2 What this reframing preserves

- The decision to use probabilistic models is preserved. Probabilistic models are well-suited to decision-making under uncertainty.
- The decision to study regime-aware models is preserved. Regimes are a reasonable hypothesis about non-stationarity.
- The decision to focus on equity factors is preserved. It is a tractable initial domain.
- The decision to build an open-source benchmark is preserved. It is the correct contribution for this kind of project.

The reframing changes **what is primitive** and **what must be justified**, not the broad domain of study.

---

## 9. The Four Fundamental Sub-Problems

The root problem decomposes into four sub-problems. They are separable in principle but coupled in practice.

### 9.1 Representation

How is the state of the world represented?

- What is observable at time `t`?
- What is latent?
- What is the state space?
- Is the state discrete, continuous, or hybrid?
- How is non-stationarity encoded?

This sub-problem determines what can be learned.

### 9.2 Inference

How is the latent state estimated from observations?

- What is the probabilistic model?
- How are parameters estimated?
- How is uncertainty over the state represented?
- How is uncertainty over parameters represented?
- How is model uncertainty represented?

This sub-problem determines what is known.

### 9.3 Decision

How is the inferred state mapped to an allocation?

- What is the objective?
- What is the utility function?
- What are the constraints?
- How are costs modeled?
- How is risk controlled?

This sub-problem determines what is done.

### 9.4 Evaluation

How is the decision process judged?

- What is the out-of-sample protocol?
- What metrics are used?
- How is statistical significance assessed?
- How is robustness tested?
- How is overfitting detected?

This sub-problem determines what is believed.

### 9.5 Why this decomposition matters

v1 collapsed all four sub-problems into a single pipeline:

> features → MoE → predictions → magnitude weights → Sharpe ratio

This made it impossible to diagnose which sub-problem was causing any observed outcome. A poor Sharpe ratio could come from bad representation, bad inference, bad decisions, or bad evaluation. There was no way to tell.

The v2 framing keeps the four sub-problems separate so that each can be developed, tested, and criticized independently.

---

## 10. Primitive Questions

These are the questions that must be answered before any modeling begins. They are stated in primitive terms.

### 10.1 About capital

- Whose capital? An individual? An institution? A fund?
- What is the risk tolerance?
- What is the time horizon?
- What is the drawdown tolerance?

### 10.2 About allocations

- What is the admissible set?
- Long-only or long-short?
- Leverage allowed?
- Position limits?
- Turnover limits?

### 10.3 About time

- What is the rebalancing frequency?
- Why that frequency?
- Does the frequency match the information arrival rate?

### 10.4 About uncertainty

- What sources of uncertainty are modeled?
- What sources are ignored?
- What is the cost of ignoring them?

### 10.5 About cost

- What is the cost model?
- Is it calibrated or assumed?
- Does it depend on turnover, size, or liquidity?

### 10.6 About non-stationarity

- Is it modeled, assumed away, or tested?
- If modeled, how?
- If assumed away, what is the justification?

### 10.7 About partial observability

- What is observed?
- What is latent?
- What is assumed observable but actually is not?

These questions feed directly into `02_problem_framing.md`.

---

## 11. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Organize the problem around primitives, not derived concepts. | Derived concepts are choices and must be justified. |
| D2 | Treat regimes as a hypothesis, not an axiom. | Non-stationarity is primitive; regimes are one model of it. |
| D3 | Separate representation, inference, decision, and evaluation. | Coupling them made v1 undiagnosable. |
| D4 | Treat v1 assumptions as auditable, not inherited. | See Section 7. |
| D5 | Require every choice to be traceable to a primitive. | Prevents reintroducing unjustified conventions. |
| D6 | Freeze this document before writing `02_problem_framing.md`. | The framing must be built on a stable decomposition. |

---

## 12. Open Questions

1. Is "non-stationarity" truly primitive, or is it a consequence of partial observability? If the full state were observed, would the process appear stationary?
2. Is "cost of action" primitive, or is it a friction introduced by market microstructure? Could a frictionless market exist in principle?
3. Is the four-sub-problem decomposition (representation, inference, decision, evaluation) complete, or is there a fifth?
4. Should "capacity" be treated as a primitive, or is it a constraint on the allocation set?
5. Does the reframing implicitly assume a single-agent investor? If so, is that a primitive or a choice?
6. Should "model uncertainty" be treated as a first-class primitive, or is it a consequence of partial observability?
7. Is there a primitive that v1 captured implicitly and this decomposition has dropped?

---

## 13. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial first-principles decomposition. |

---

## 14. References

- `00_docs_index.md` — documentation map and conventions.
- v1 `problem_framing.md` — reference for inherited assumptions.
- v1 `README.md` — reference for v1 claims and results.
- Fama & French (1993) — factor model origins.
- Ang & Bekaert (2002) — regime-switching in asset allocation.
- Jacobs et al. (1991) — Mixture of Experts.
