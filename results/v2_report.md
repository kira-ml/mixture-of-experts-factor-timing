# moe-factor-timing v2 — Results Report

**Project:** Decision-Focused Regime-Aware Factor Allocation
**Version:** 2.0
**Status:** Final draft (Phase 5)
**Date:** 2026-09-24
**Reference run:** `runs/20260923T173318Z`
**Robustness runs:** 6 configurations, `results/robustness_sweep.csv`
**λ sweep:** 4 configurations, `results/lambda_sweep.csv`

---

## 1. Executive Summary

We evaluate whether probabilistic regime-aware models — ridge regression
and a two-state Gaussian Hidden Markov Model — produce better
out-of-sample allocation decisions than a naive equal-weight (1/N)
benchmark under realistic costs and constraints.

**Result: they do not.** In 6/6 robustness configurations, 1/N beats
both probabilistic models on expected utility after costs. The
underperformance is not a risk-aversion artifact (a λ sweep from 2 to 20
does not close the gap), not a solver artifact (fallback rate = 0.0
across all models and rebalances), and not a feature artifact (V13
leakage check clean).

Both probabilistic models produce miscalibrated predictive
distributions: ridge is biased (PIT median 0.63), HMM is overconfident
(14% and 18% of PIT values fall in the lower and upper 5% tails
respectively, vs 5% each expected). `07` Section 8.1 **S5** fires.
Predictive metrics (NLL, CRPS) are downgraded to exploratory.

**Secondary finding.** In the single configuration where HMM passes PIT
calibration (`min_train=132, K=3`, p=0.089), its expected utility is
still 45% below 1/N. **A calibrated predictive distribution did not
translate into decision value.**

---

## 2. Problem Statement

Given finite capital, a set of tradable US equity factor ETFs, monthly
rebalancing, and non-zero costs, does explicitly modeling latent state
and model uncertainty improve out-of-sample allocation decisions,
measured by expected utility after costs and validated with statistical
discipline?

Primary research question: `02_problem_framing.md` Section 7.1.

---

## 3. Setup

### 3.1 Universe

| ID | Ticker | Asset class | Role |
|---|---|---|---|
| SPY | SPY | US large-cap equity | Market |
| IWD | IWD | US large-cap value | Value |
| MTUM | MTUM | US large-cap momentum | Momentum |
| QUAL | QUAL | US large-cap quality | Quality |
| USMV | USMV | US minimum volatility | Low Volatility |
| CASH | BIL | 1–3 month T-bills | Risk-free residual |

Universe fixed before evaluation. VIX excluded from allocation, used as feature only.

### 3.2 Features

Seventeen features, all lagged at least one month:

- Fifteen lagged returns: 5 assets × lags {1, 6, 12}
- Two macro: `T10Y2Y`, `VIXCLS` (FRED latest + 1-month lag; ALFRED vintages deferred)

`T / p = 96 / 17 = 5.6`, satisfies `06` Section 7.4 rule of thumb.

### 3.3 Models

| Model | Type | Notes |
|---|---|---|
| naive_1n | Fixed | Equal weight |
| cash | Fixed | 100% BIL |
| persistence | Fixed | Long positive last-month return |
| rolling_avg | Fixed | 12-month mean |
| momentum | Fixed | EWMA (decay 0.9, window 12) |
| ridge | Probabilistic | α=1.0, Gaussian residual covariance |
| hmm_gaussian | Probabilistic | K=2, diagonal covariance, 100 EM iterations |

### 3.4 Decision rule

Constrained mean-variance:

```
w_t* = argmax_{w ∈ W}  w^T μ_t − (λ/2)·w^T Σ_t w − C(w, w_{t-1})
```

with constraints `w ≥ 0`, `Σw ≤ 1`, `w_i ≤ 0.40`, `||w − w_{t-1}||_1 ≤ 0.50`.
Cost model: 10 bps assumed per unit turnover. λ = 10 (sensitivity reported).

### 3.5 Split

Expanding walk-forward, `min_train=96`, test step 1 month, no shuffling.
Default out-of-sample window: **49 months** (limited by 2013-08-01
start date, driven by ETF inception dates for MTUM/QUAL/USMV).

---

## 4. Evaluation Protocol

Following `07`:

| Layer | Metric | Role |
|---|---|---|
| L1 — Predictive | NLL | Primary predictive (`07` 5.6) |
| L1 — Predictive | CRPS, PIT, KS | Calibration checks |
| L2 — Point | RMSE, MAE | Diagnostics only |
| L3 — Decision | **Expected utility** | **Primary decision** (`07` 7.5) |
| L3 — Decision | Sharpe, Sortino, MaxDD, Calmar, turnover, cost drag | Complements |
| L4 — Statistical | Bootstrap CIs (1000 reps, block length 3) | Required |
| L5 — Robustness | Distribution across `min_train` × `K` | Required |

**Model selection separated from evaluation** (`07` Section 10.6).
All hyperparameters fixed a priori. Evaluation window never used for
selection.

---

## 5. Results

### 5.1 Baseline (default config)

| Model | Expected utility | 95% CI | Sharpe | Ann ret | Ann vol | Turnover |
|---|---:|---|---:|---:|---:|---:|
| persistence | **0.00719** | [−0.0018, 0.0159] | 1.315 | 18.7% | 13.9% | 13.36 |
| naive_1n | **0.00689** | [−0.0014, 0.0152] | 1.287 | 17.6% | 13.4% | 0.50 |
| rolling_avg | 0.00679 | [−0.0013, 0.0155] | 1.283 | 18.5% | 14.1% | 5.10 |
| momentum | 0.00648 | [−0.0019, 0.0152] | 1.255 | 18.0% | 14.0% | 6.28 |
| ridge | 0.00487 | [0.0001, 0.0091] | 1.123 | 9.2% | 8.2% | 5.81 |
| hmm_gaussian | 0.00430 | [−0.0032, 0.0113] | 1.016 | 10.3% | 10.2% | 6.50 |
| cash | 0.00359 | [0.0032, 0.0039] | 17.2 | 4.4% | 0.25% | 0.0 |

Ridge trails 1/N by 29% on expected utility. HMM trails by 38%.

### 5.2 Robustness distribution (6 configurations)

Expected utility by `min_train` × `K`:

| min_train | K | 1/N | Ridge | HMM | 1/N wins |
|---:|---:|---:|---:|---:|:---:|
| 60 | 2 | 0.001652 | 0.000698 | 0.000348 | ✅ |
| 60 | 3 | 0.001652 | 0.000698 | 0.000745 | ✅ |
| 96 | 2 | 0.006914 | 0.004894 | 0.004333 | ✅ |
| 96 | 3 | 0.006913 | 0.004894 | 0.004467 | ✅ |
| 132 | 2 | 0.009663 | 0.004524 | 0.005744 | ✅ |
| 132 | 3 | 0.009663 | 0.004524 | 0.005319 | ✅ |

**6/6: 1/N dominates.** Robust across window length and number of regimes.

### 5.3 λ sensitivity

| λ | Ridge EU | Ridge Sharpe | Ridge cash | HMM EU | HMM Sharpe | HMM cash |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | — | 1.236 | 25.7% | — | 0.998 | 23.5% |
| 5 | — | 1.175 | 36.2% | — | 1.019 | 23.5% |
| **10** | **0.00487** | **1.126** | **57.6%** | **0.00430** | **1.020** | **23.5%** |
| 20 | — | 1.546 | 78.1% | — | 1.042 | 23.5% |

At λ=2, ridge is 25.7% cash (near fully invested) and still trails 1/N
by 0.05 Sharpe. **λ does not explain the underperformance.** HMM's cash
residual is invariant to λ (23.47% ± 0.0002 across four values),
indicating its allocation is dictated by the constraint set, not by the
objective.

### 5.4 Predictive layer (exploratory, S5 fired)

| Model | NLL | CRPS | PIT KS stat | PIT KS p | Failure mode |
|---|---:|---:|---:|---:|---|
| ridge | −11.41 | 0.0247 | 0.1368 | 1.84e-4 | **Bias** (median PIT 0.63) |
| hmm_gaussian | −5.90 | 0.0336 | 0.1530 | 1.80e-5 | **Overconfidence** (tails 3×) |

Both models fail `07` Section 8.1 **S5**. NLL and CRPS rankings are
non-confirmatory.

### 5.5 Calibration vs decision value

HMM at `min_train=132, K=3` passes PIT (p=0.089). Its expected utility
in that configuration is 0.005319 vs 1/N's 0.009663 — **45% below**.
A calibrated predictive distribution did not translate into decision
value in this setup.

### 5.6 Solver reliability

Fallback rate = **0.0** for every model across all 49 rebalances.
`09` Section 5.1 T6 does not fire. The negative result is not a solver
artifact.

### 5.7 Reproducibility

Determinism check (`scripts/check_determinism.py`): **18/18 result
artifacts byte-identical across two independent runs** on a reduced
config. Full-config runs show ≤0.005 Sharpe drift for probabilistic
models (immaterial to ranking). See `docs/11_phase1_validation_log.md`.

---

## 6. Criteria Fired

Per `09`:

| ID | Criterion | Evidence | Action taken |
|---|---|---|---|
| **E1** | No model beats 1/N on EU | 6/6 configs | Stop. Report negative result. |
| **E4** | Fails in ≥4 of 9 robustness dimensions | 6/6 tested dims | Report as not robust. |
| **EK1** | After MVP + 2 iterations, no model wins | Iteration budget exhausted | Kill regime-aware direction. |
| **EK3** | Decision-focused optimization never beats EW | Direct restatement of E1 | Kill decision-focused direction. |
| **S4** | Bootstrap CIs wider than effect size | EU CIs span ±0.007 vs effects ~0.002 | Report as inconclusive. |
| **S5** | PIT calibration fails for best model | Ridge KS p=1.84e-4 | Predictive metrics downgraded to exploratory. |
| **SK1** | Sample insufficient for confirmatory claims | 49-month OOS window | Report as exploratory, not confirmatory. |
| 13.2 | Kill decision-focused approach | After MVP + 2 iterations, no beat of 1/N | Report negative finding. |

## 7. Criteria Not Tested

The following criteria are referenced by `07` but were not evaluated in
this MVP. They are disclosed, not silently omitted:

| ID | Criterion | Reason not tested |
|---|---|---|
| E3 | Deflated Sharpe ratio | Not implemented |
| E7 | Subperiod dependence | Not run |
| E8 | Regime stability across seeds | Not run |
| E9 | Regime-conditional returns distinguishable | Not run |
| S1 | Minimum detectable Sharpe > 1.0 | Power analysis not computed |
| S2 | MinTRL > OOS window | Not computed |
| S3 | N_trials exceeds DSR correction | DSR not computed |
| SK3 | DSR not computable | N/A — DSR not implemented |

These omissions are a limitation of the MVP, not evidence in either
direction. The negative result stands on E1, E4, and EK1, which do not
depend on the untested criteria.

---

## 8. Non-Goals (verbatim from `03_scope_and_non_goals.md` Section 9)

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

---

## 9. Number of Configurations Tested

`N_trials = 22` (11 configurations × 2 probabilistic models):

- 1 baseline
- 4 λ sweep
- 6 robustness sweep (min_train × K)

See `runs/<run_id>/report/n_trials.csv` and `docs/10_experiment_log.md`.

Multiple-testing correction (DSR) was **not** applied. All reported
results are therefore **exploratory** with respect to selection. This
disclosure is required by `07` Section 8.6.

---

## 10. Capacity

No market-impact model was calibrated. Capacity is reported as
`not_estimated` (`runs/<run_id>/report/capacity.csv`). See `05` Section
7.3 and `08` Section 13.1.

---

## 11. Limitations

1. **Short out-of-sample window.** 49 monthly periods. Minimum detectable
   Sharpe at 95% confidence is approximately 0.4; sharper comparisons
   between models are underpowered. `09` Section 8.2 SK1 fires.

2. **Underpowered for Sharpe claims.** Distinguishing Sharpe 1.29 from
   1.13 at 95% confidence requires ~300 years of data.

3. **Sub-0.01 Sharpe run-to-run drift** for probabilistic models.
   Immaterial to rankings; disclosed per `04` Section 10.4 IV.4.

4. **FRED latest + 1-month lag** in place of ALFRED vintages. Conservative
   approximation; documented in `06` Section 6.5.

5. **Two utility functions only.** Mean-variance was evaluated; CRRA and
   CVaR were not, per `09` Section 7.5 iteration budget.

6. **Cost model assumed at 10 bps.** Calibrated costs were not tested.

7. **Single asset class.** US equity factors only. Conclusions do not
   extend to other classes.

8. **Single rebalancing frequency.** Monthly only.

---

## 12. Conclusion

Under the tested configurations, no probabilistic regime-aware model
produced out-of-sample decision value above the equal-weight benchmark.
The result is robust across training window length, number of latent
states, and risk-aversion parameter. It survives a fully validated
pipeline (determinism, solver fallback, leakage detection).

The secondary finding — that a calibrated predictive distribution does
not imply decision value — is the more interesting result. It suggests
that the current focus in the regime-aware factor allocation literature
on predictive calibration may be insufficient for decision improvement.

The benchmark's value is its rigor, not its returns. v2 answers the
research question in `02` with the evidence available and reports
honestly that the evidence is insufficient to distinguish the models
from chance.

---

## 13. Reproducibility

- **Reference run:** `runs/20260923T173318Z/`
- **Manifest:** SHA-256 hashes of every artifact, config hash, code commit
- **Determinism:** 18/18 artifacts reproducible (`scripts/check_determinism.py`)
- **Config:** `configs/default.yaml`
- **Data snapshot:** `runs/20260923T173318Z/raw/`
- **Validation log:** `docs/11_phase1_validation_log.md`
- **Experiment log:** `docs/10_experiment_log.md`

Reproduce with:

```bash
python -m src.run --config configs/default.yaml
```

---

## 14. References

- `docs/00_docs_index.md` through `docs/11_phase1_validation_log.md`
- `results/robustness_sweep.csv`, `results/lambda_sweep.csv`
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive
  Diversification. *Review of Financial Studies*, 22(5), 1915–1953.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio.
  *Journal of Portfolio Management*, 40(5), 94–107.
