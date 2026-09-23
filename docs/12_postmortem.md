# Postmortem

**Project:** moe-factor-timing v2
**Status:** Frozen
**Last Updated:** 2026-09-24

---

## 1. Purpose

Closing document for v2. Records what was built, what was found,
what the process taught, and what a v3 would need to change.

Not a summary of `results/v2_report.md`. The postmortem is a
projection forward: what the next project needs to know that is not
in the report.

---

## 2. Context and Dependencies

- **Depends on:** `09_stop_criteria_and_kill_criteria.md` Section 13.5,
  `results/v2_report.md`, `docs/10_experiment_log.md`,
  `docs/11_phase1_validation_log.md`
- **Feeds into:** any future v3 pre-code documentation set
- **Reference only:** v1 archived under `archive/v1/`, tag `v1.0-final`

---

## 3. Outcome

v2 tested whether probabilistic regime-aware models improve
out-of-sample factor allocation decisions versus a naive equal-weight
benchmark, under realistic costs and constraints.

**They do not.** In 6/6 robustness configurations, 1/N beats both
ridge regression and a two-state Gaussian HMM on the primary decision
metric (expected utility after costs). The result is robust across
training-window length, number of latent states, and risk-aversion
parameter. It survives a fully validated pipeline:

- Determinism: 18/18 result artifacts byte-identical across two runs
- Solver fallback rate: 0.0 across 49 rebalances × 7 models
- V13 leakage check: fires on injected leakage, silent on legitimate features
- Shape assertions: X/y alignment verified every rebalance

**Secondary finding:** calibration does not imply decision value.
The HMM passes PIT calibration in one configuration
(`min_train=132, K=3`, KS p=0.089) and still trails 1/N by 45% on
expected utility in that configuration.

---

## 4. What went right

### 4.1 The pre-committed criteria worked

`docs/09` was written before any result was seen. It prevented three
specific overclaiming paths:

1. **λ = 20 artifact.** Ridge Sharpe at λ=20 was 1.546 — higher than
   1/N's 1.29. Reporting it would have claimed outperformance. But
   78% cash is not a decision; the criteria forced the exposure
   diagnostic that exposed it.

2. **Calibration-as-success.** HMM passed PIT at one configuration. It
   would have been tempting to headline that as a positive predictive
   result. `09` Section 7.1 E1 required also beating 1/N on expected
   utility, which HMM did not.

3. **Iteration creep.** After seeing ridge at 1.13 Sharpe, the
   natural pull was "one more α tweak, one more K, one more feature."
   The 3-iteration budget (`09` Section 7.5) made that impossible.
   The project stopped when the criteria said to.

### 4.2 The pipeline produced a trustworthy negative result

Every headline number is traceable to a config + raw data snapshot
(SHA-256 manifest). Determinism check: 18/18 artifacts identical.
Falsification test: V13 fires on injected leakage. Solver fallback:
0.0 across all 49 rebalances × 7 models. The negative result is not
an artifact.

### 4.3 Two real bugs were caught before any conclusion

- Cash residual earned zero, not risk-free. Sharpe 0.00 exposed it.
- HMM covariance rank-1 collapse. NLL 866 exposed it.

Both fixed, re-run, verified. The pipeline runs cleanly end-to-end
on the frozen configuration.

### 4.4 The secondary finding is more interesting than the primary

The primary result is a negative. The secondary finding — that a
calibrated predictive distribution does not translate into decision
value — is a positive contribution to the field's understanding of
the predict-then-optimize separation. It survived the pipeline
validation and is not an artifact of solver or feature construction.

---

## 5. What went wrong

### 5.1 Phase 1 exit gate was skipped

`09` Section 10.2 lists five Phase 1 exit requirements. Three
(contract tests, falsification tests, integration test) were not
satisfied when Phases 2–4 experiments ran. They were closed
retroactively — `docs/11_phase1_validation_log.md`.

This is the most important process lesson. **The gate must be closed
before downstream work is authorized, not after.** If a bug had been
found in the retroactive validation, every experiment from the
baseline onward would have been invalidated.

The retroactive closure worked because no bug was found. It was luck,
not discipline. A v3 cannot rely on the same luck.

### 5.2 Full-config reproducibility drift

Determinism check passes on a reduced config. Full-config runs show
≤0.005 Sharpe drift for probabilistic models. Cause: OSQP
nondeterminism despite `polish=True, threads=1`. Immaterial to
conclusions, but violates `04` invariant IV.4 in the strict sense.
Should be fixed for v3 (switch to CLARABEL or another strictly
deterministic solver).

### 5.3 FRED vintages approximated

ALFRED vintage loading was deferred. FRED latest + 1-month lag
substituted. Conservative in the direction of understating model
performance, but not the contract specified in `06` Section 6.2.

### 5.4 HMM convergence warnings never silenced

`hmmlearn` prints `Model is not converging` on nearly every fit.
Log-likelihood moves by ~0.001% per iteration; the model has
effectively converged. Cosmetic, but pollutes terminal output across
every run. The one-line fix (`tol=1e-2` in the `GaussianHMM`
constructor) was identified but never applied. Add it to v3's
pipeline from day one.

---

## 6. What was not tested

Eight criteria referenced in `07` were not evaluated. Each is a v3
decision: test or explicitly defer.

| ID | Criterion | Reason not tested |
|---|---|---|
| E3 | Deflated Sharpe ratio | Not implemented |
| E7 | Subperiod dependence | Not run |
| E8 | Regime stability across seeds | Not run |
| E9 | Regime-conditional returns distinguishable | Not run |
| S1 | Minimum detectable Sharpe > 1.0 | Power analysis not computed |
| S2 | MinTRL > OOS window | Not computed |
| S3 | N_trials exceeds DSR correction | DSR not implemented |
| SK3 | DSR not computable | N/A |

These omissions do not weaken the negative result, which rests on E1,
E4, and EK1 — none of which depend on the untested criteria. But they
would need to be addressed for any future positive claim.

Additional untested items from `07`:

- Cross-asset correlation dynamics
- Regime transition matrix interpretation
- Sensitivity to calibrated (vs assumed) costs
- Sensitivity to utility function (CRRA, CVaR)

---

## 7. What v3 would need

### 7.1 A longer sample

49 out-of-sample months is underpowered. Sharpe CIs span ±0.8 for
everything. Either extend history with academic factor data
(Fama–French, AQR) back to the 1980s, or accept an exploratory-only
project with no confirmatory claims.

### 7.2 A different hypothesis

The regime hypothesis on five US equity ETFs is not obviously
supportable. Three candidate directions:

1. **Stronger regime dependence.** Test on assets where regimes are
   more pronounced — credit spreads, commodity futures, VIX futures.
2. **Regime-conditional risk targeting.** Instead of
   regime-conditional expected returns (the v2 hypothesis), test
   regime-conditional volatility targeting. More likely to survive
   because volatility is more predictable than returns.
3. **Decision-focused training.** Train the model on realized utility
   rather than NLL, bypassing the two-stage predict-then-optimize
   separation. The v2 finding that calibration and decision quality
   diverge suggests this is where the loss is.

### 7.3 A pre-committed Phase 1 exit gate

`09` for v3 must make the Phase 1 gate binding before any experiment
runs. Contract tests, falsification tests, determinism check, and
fallback-rate reporting must all pass. No exceptions.

### 7.4 A strictly deterministic solver

Replace OSQP with CLARABEL, or verify that OSQP with `polish=True,
threads=1` produces byte-identical output at full config (not just
at reduced config). The determinism check must be run at full config
in v3, not at a reduced config.

### 7.5 A real vintage data loader

ALFRED vintages, not FRED latest + lag. Or an explicit decision to
abandon macro features entirely if vintages are not available.

### 7.6 What v3 should not do

- Do not re-run the same hypothesis on the same universe with more
  features, more models, or more hyperparameters. v2 already answered
  that question negatively.
- Do not shorten the sample.
- Do not relax the iteration budget.
- Do not begin coding before `01`–`09` are frozen.
- Do not defer Phase 1 validation to Phase 4.

---

## 8. Value delivered

Even as a negative result, v2 produced:

1. A reproducible ten-stage pipeline with contracts at every boundary.
2. A pre-committed protocol that survived the temptation to overfit.
3. A quantified negative result with all required disclosures.
4. A secondary finding — calibration ≠ decision value — that is more
   interesting than the primary result.
5. An audit trail: nine experiments, six validations, every run hashed.

The benchmark's value is its rigor. That is the deliverable.

---

## 9. Ties to Other Documents

| Concern | Where specified |
|---|---|
| Primary result | `results/v2_report.md` |
| Experiment reasoning | `docs/10_experiment_log.md` |
| Validations | `docs/11_phase1_validation_log.md` |
| Criteria and phase gates | `docs/09_stop_criteria_and_kill_criteria.md` |
| Reporting standard | `docs/07_evaluation_framework_and_backtesting_protocol.md` Section 12 |
| Data contract | `docs/06_data_information_and_target_contract.md` |
| API contracts | `docs/16_api_contracts_and_interfaces.md` |

---

## 10. Open Questions

1. Should v3 be started at all, or is the negative result sufficient
   to retire the research direction?
2. If v3 is started, should the pipeline be reused (inheriting the
   ≤0.005 Sharpe drift) or rebuilt (a third implementation)?
3. Is the 3-iteration budget too tight or too loose for a v3 with a
   longer sample?
4. Should the retroactive Phase 1 closure be treated as a process
   failure worth documenting in the v3 pre-code set, or as an
   isolated incident?
5. Is the "calibration ≠ decision value" finding strong enough to
   motivate a standalone paper, independent of v3?
6. Should v3 target US equity factors again with a different
   hypothesis, or move to a different asset class entirely?

---

## 11. Version History

| Version | Date | Changes |
|---|---|---|
| 0.1 | 2026-09-24 | Initial postmortem at v2 release. |

---

## 12. References

- `results/v2_report.md`
- `docs/10_experiment_log.md`
- `docs/11_phase1_validation_log.md`
- `docs/09_stop_criteria_and_kill_criteria.md` Section 13.5
- `docs/16_api_contracts_and_interfaces.md`
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus
  Naive Diversification. *Review of Financial Studies*, 22(5),
  1915–1953.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe
  Ratio. *Journal of Portfolio Management*, 40(5), 94–107.
