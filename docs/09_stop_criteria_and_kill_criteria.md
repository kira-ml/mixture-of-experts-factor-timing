# Stop Criteria and Kill Criteria

**Project:** moe-factor-timing v2
**Status:** Draft
**Last Updated:** 2026-08-03

---

## 1. Purpose

This document defines, **before any code is written and before any result is seen**, the conditions under which the project — or a specific model, hypothesis, or direction — must stop, pivot, or be killed.

Its job is to prevent five failure modes:

1. **Sunk-cost continuation.** Continuing past the point where evidence says stop, because effort has already been invested.
2. **Goalpost moving.** Redefining success after seeing results so that the project can claim victory.
3. **Endless iteration.** Never declaring a direction finished because no stopping rule exists.
4. **Overclaiming under weak evidence.** Reporting a positive result that fails the pre-committed validity bars.
5. **Silent scope creep.** Expanding the project to cover new questions instead of resolving the existing one.

The core rule:

> **The criteria in this document are committed to before results are seen. Once a criterion is met, the corresponding action is taken. The criteria are not renegotiated after the fact.**

If a criterion turns out to be wrong, it is wrong in the record. It is not retroactively softened to accommodate an inconvenient result.

---

## 2. Context and Dependencies

- **Depends on:** `00_docs_index.md`, `01_first_principles_problem_decomposition.md`, `02_problem_framing.md`, `03_scope_and_non_goals.md`, `04_assumptions_and_invariants.md`, `05_objective_utility_and_decision_specification.md`, `06_data_information_and_target_contract.md`, `07_evaluation_framework_and_backtesting_protocol.md`, `08_minimum_viable_baseline_end_to_end_pipeline_architecture.md`
- **Feeds into:** project execution, phase gates, and the final report
- **Reference only:** v1 `TODO.md` (for the pattern of stopping decisions in practice)

This document is the last of the pre-code set. After this is frozen, implementation begins.

---

## 3. Philosophy

### 3.1 Success is not "the model wins"

The v1 framing implicitly treated project success as "MoE beats baselines." That framing is rejected.

In v2, project success means:

> **A rigorous, reproducible benchmark was produced that answers the primary research question with the evidence available, and that reports honestly when evidence is insufficient.**

A project that produces a well-documented negative result is a successful project. A project that produces a positive result by violating a criterion is a failed project.

### 3.2 Criteria are procedural, not aspirational

Every criterion in this document is:

- **Binary.** Pass or fail. No "mostly passed."
- **Measurable.** Traceable to a metric, check, or artifact from `04`, `06`, or `07`.
- **Pre-committed.** Written before any evaluation runs.
- **Binding.** Met criterion → action taken, no exceptions.

### 3.3 Four kinds of decisions

The criteria produce four kinds of decisions:

| Decision | Meaning |
|----------|---------|
| **Continue** | Proceed to the next phase or next configuration. |
| **Stop (pause)** | Halt current direction, document why, and decide whether to pivot. |
| **Pivot** | Change approach (model class, feature set, utility, scope) with a documented rationale. |
| **Kill** | Abandon the current hypothesis, direction, or the project itself. |

"Stop" is not the same as "kill." Stop means the current approach is not working. Kill means the question itself or the project is not viable.

### 3.4 No silent extensions

Any decision to relax a criterion — even slightly — requires:

1. A written justification in `docs/10_decision_log.md` (to be created).
2. A note in the final report.
3. An acknowledgment that the relaxed result is **exploratory**, not confirmatory.

A relaxed criterion never produces a confirmatory claim.

---

## 4. Categories of Criteria

Criteria are organized by what they protect:

| Category | Protects against | Section |
|----------|------------------|---------|
| **Technical** | Broken pipelines, invalid runs | 5 |
| **Methodological** | Contracts being silently violated | 6 |
| **Empirical** | Continuing past the point where models add value | 7 |
| **Statistical** | Overclaiming on underpowered evidence | 8 |
| **Scope** | Drift into new questions | 9 |
| **Phase gates** | Proceeding without prerequisites | 10 |

Each category has stop criteria and kill criteria.

---

## 5. Technical Stop Criteria

Technical criteria trigger when the pipeline cannot produce a valid run.

### 5.1 Stop criteria

| ID | Condition | Action |
|----|-----------|--------|
| T1 | Ingest fails for any required series after 3 documented retries | Stop. Diagnose source. |
| T2 | Any validation check (V1–V21) fails | Stop. Abort run. |
| T3 | Feature construction produces NaN at rate > 5% for any feature | Stop. Fix or drop feature. |
| T4 | Training window smaller than `min_train` for any test period | Stop. Adjust split. |
| T5 | Any model fails to fit on the training window | Stop. Diagnose model. |
| T6 | Solver fails to converge on > 5% of rebalances | Stop. Revisit decision rule. |
| T7 | Run manifest incomplete or hashes mismatch | Stop. Invalidate run. |
| T8 | Reproducibility check (re-run with same config) produces different hashes | Stop. Fix non-determinism. |

### 5.2 Kill criteria

| ID | Condition | Action |
|----|-----------|--------|
| TK1 | Required data source is permanently unavailable | Kill the run; reassess universe. |
| TK2 | Vintage data unavailable for any macro feature and no acceptable lag substitute | Kill the macro feature set. |
| TK3 | Solver cannot be made deterministic under fixed seeds | Kill the decision rule implementation; replace solver. |
| TK4 | Any required invariant (`04`, Section 10) cannot be enforced programmatically | Kill the run design; reassess architecture. |

### 5.3 Continue criteria

Continue only if T1–T8 all pass and TK1–TK4 do not apply.

---

## 6. Methodological Stop Criteria

Methodological criteria trigger when the contracts in `04`, `06`, and `07` cannot be respected.

### 6.1 Stop criteria

| ID | Condition | Action |
|----|-----------|--------|
| M1 | Information set cannot be made vintage-consistent | Stop. Redefine features. |
| M2 | Feature matrix cannot satisfy the `T / p ≥ 5` rule | Stop. Reduce feature set. |
| M3 | Decision rule cannot be derived from the objective | Stop. Revisit decision spec. |
| M4 | Benchmark cannot be run under the same protocol as models | Stop. Redefine benchmark. |
| M5 | Bootstrap CI cannot be computed for a headline metric | Stop. Report the limitation or exclude metric. |
| M6 | Multiple-testing correction cannot be applied to the tested configurations | Stop. Reduce trials or use a documented alternative. |

### 6.2 Kill criteria

| ID | Condition | Action |
|----|-----------|--------|
| MK1 | Decision rule cannot be specified without an ad hoc allocation | Kill the current decision framework. |
| MK2 | Model selection cannot be separated from evaluation | Kill the evaluation protocol. |
| MK3 | Feature contract cannot be enforced programmatically | Kill the feature set. |
| MK4 | Invariants `I.1`–`I.6` cannot all be enforced | Kill the pipeline design. |

### 6.3 Continue criteria

Continue only if M1–M6 all pass and MK1–MK4 do not apply.

---

## 7. Empirical Stop Criteria

Empirical criteria trigger when models provide no decision value after costs, or when the primary metric is not distinguishable from a benchmark.

### 7.1 Primary empirical criterion

**E1 — No model produces out-of-sample expected utility above 1/N after costs.**
Action: Stop. The benchmark is a valid negative result. Report it.

### 7.2 Secondary empirical criteria

| ID | Condition | Action |
|----|-----------|--------|
| E2 | Best model's Sharpe CI includes zero | Stop. Report as "not distinguishable from chance." |
| E3 | Best model's deflated Sharpe ratio p-value > 0.05 | Stop. No significance claim permitted. |
| E4 | Best model fails to beat 1/N in 4 out of 9 robustness dimensions | Stop. Result is not robust. |
| E5 | Turnover-adjusted returns are negative for all models | Stop. Costs dominate signal. |
| E6 | Cost drag exceeds gross return for the best model | Stop. Strategy is not implementable. |
| E7 | Best model's results depend on a single subperiod | Stop. Report instability. |
| E8 | Latent regime assignments are not stable across seeds | Stop. Regime hypothesis is not supported. |
| E9 | Regime-conditional returns are not statistically distinguishable across regimes | Stop. Regimes are not economically distinct. |

### 7.3 Kill criteria

| ID | Condition | Action |
|----|-----------|--------|
| EK1 | After MVP + 2 documented iterations, no model beats 1/N after costs | Kill the regime-aware direction. |
| EK2 | K > 1 never improves out-of-sample utility over K = 1 | Kill the regime hypothesis. |
| EK3 | Decision-focused optimization never beats equal-weight | Kill the decision-focused direction. |
| EK4 | All performance is explained by a single factor tilt | Kill the "regime-aware" framing; report as factor exposure. |

### 7.4 Continue criteria

Continue only if:

- E1 passes (some model beats 1/N after costs), **and**
- At least 4 of E2–E9 do not fire, **and**
- No EK1–EK4 apply.

### 7.5 Iteration budget

- MVP: 1 iteration.
- Post-MVP refinements: **maximum 2 iterations**.
- If after 3 total iterations no criterion in 7.1–7.4 produces a clean "continue," the project pivots or stops.

The iteration budget is binding. It prevents endless refinement.

---

## 8. Statistical Stop Criteria

Statistical criteria trigger when the sample cannot support the claims the project intends to make.

### 8.1 Stop criteria

| ID | Condition | Action |
|----|-----------|--------|
| S1 | Minimum detectable Sharpe ratio > 1.0 for the planned sample | Stop making Sharpe claims. Use utility and calibration. |
| S2 | Minimum track record length > available out-of-sample window | Stop making significance claims. |
| S3 | Number of tested configurations exceeds what DSR can correct for | Stop. Report as exploratory. |
| S4 | Bootstrap CIs are wider than the effect size being claimed | Stop. Report as inconclusive. |
| S5 | PIT calibration fails for the best model | Stop. Predictive distribution is invalid. |
| S6 | NLL ranking contradicts decision ranking with no explanation | Stop. Diagnose or report both. |

### 8.2 Kill criteria

| ID | Condition | Action |
|----|-----------|--------|
| SK1 | Sample size is insufficient for any confirmatory claim, and extending it is out of scope | Kill the confirmatory framing. Report as exploratory only. |
| SK2 | Multiple-testing correction nullifies all reported improvements | Kill the "outperformance" claim. |
| SK3 | Deflated Sharpe ratio is not computable under the actual `N_trials` | Kill the Sharpe claim. |

### 8.3 Continue criteria

Continue only if S1–S6 all pass and SK1–SK3 do not apply.

If SK1 applies, the project **continues as exploratory**, not confirmatory. This is a valid outcome.

---

## 9. Scope Stop Criteria

Scope criteria trigger when the project drifts from its defined boundaries.

### 9.1 Stop criteria

| ID | Condition | Action |
|----|-----------|--------|
| SC1 | A proposed addition does not answer a research question in `02` | Stop. Reject the addition. |
| SC2 | A proposed addition is out of scope per `03`, Section 5 | Stop. Reject or defer. |
| SC3 | A proposed addition would require rerunning an evaluation that has already frozen | Stop. Defer to a new run. |
| SC4 | A proposed addition changes the decision rule | Stop. Restart evaluation from scratch. |
| SC5 | A proposed addition changes the benchmark set | Stop. Restart evaluation from scratch. |
| SC6 | A proposed addition expands the universe | Stop. Defer to a new version. |

### 9.2 Kill criteria

| ID | Condition | Action |
|----|-----------|--------|
| SCK1 | The primary research question cannot be answered within scope | Kill the project scope; redefine or retire the question. |
| SCK2 | The MVP cannot be completed within the defined module boundaries | Kill the MVP design; reassess architecture. |
| SCK3 | The project has been redirected more than twice | Kill the current direction; freeze and report. |

### 9.3 Continue criteria

Continue only if SC1–SC6 do not fire and SCK1–SCK3 do not apply.

---

## 10. Phase Gates

The project proceeds through six phases. Each phase has an entry gate and an exit gate. Passing the exit gate is required to enter the next phase.

### 10.1 Phase 0 — Documentation (current phase)

**Entry:** Project initiated.

**Exit gate:**
- `docs/00`–`docs/09` written and frozen.
- Pre-commit hooks configured (deferred).
- v1 archive complete and tagged `v1.0-final`.

**Failure action:** Do not proceed to code.

### 10.2 Phase 1 — Pipeline MVP

**Entry:** Phase 0 exit gate passed.

**Exit gate:**
- Stages 1–10 implemented per `08`, Section 7.
- Contract tests pass (`08`, Section 17.1).
- Falsification tests pass (`08`, Section 17.2).
- Integration test passes (`08`, Section 17.3).
- One full run on the default configuration produces a valid manifest.

**Failure action:** Do not proceed to evaluation.

### 10.3 Phase 2 — Baseline Evaluation

**Entry:** Phase 1 exit gate passed.

**Exit gate:**
- Baselines (1/N, persistence, rolling avg, momentum) evaluated.
- Ridge evaluated.
- One regime-aware model evaluated.
- Results pass technical and methodological criteria (Sections 5–6).

**Failure action:** Diagnose. Do not proceed to robustness.

### 10.4 Phase 3 — Primary Evaluation

**Entry:** Phase 2 exit gate passed.

**Exit gate:**
- Empirical criteria evaluated (Section 7).
- Statistical criteria evaluated (Section 8).
- No kill criteria fired.

**Failure action:**
- If E1 fails → stop, report negative result.
- If SK1 applies → continue as exploratory only.
- If EK1–EK4 fire → pivot or stop.

### 10.5 Phase 4 — Robustness

**Entry:** Phase 3 exit gate passed.

**Exit gate:**
- Robustness across declared dimensions completed (`07`, Section 9).
- Results reported as distributions, not best configurations.

**Failure action:**
- If E4 fires → stop. Report as not robust.

### 10.6 Phase 5 — Report

**Entry:** Phase 4 exit gate passed.

**Exit gate:**
- Report complies with `07`, Section 12.
- Disclosures (`03`, Section 9) present verbatim.
- Manifest and hashes complete.
- Reproducibility verified by second run.

**Failure action:** Report is invalid until compliant.

### 10.7 Phase 6 — Release

**Entry:** Phase 5 exit gate passed.

**Exit gate:**
- Results committed to `results/`.
- Repository tagged `v2.0`.
- Public README and docs consistent.

**Failure action:** Do not release until compliant.

---

## 11. Decision Tree

When a phase gate is not passed, the following tree is used.

```
Phase gate fails
│
├─► Is the failure technical? (Section 5)
│     └─► Fix and retry. If unfixable, kill run.
│
├─► Is the failure methodological? (Section 6)
│     └─► Revisit contract. If unsalvageable, kill protocol.
│
├─► Is the failure empirical? (Section 7)
│     │
│     ├─► Is it E1 (no model beats 1/N)?
│     │     └─► Stop. Report negative result. Do not iterate further.
│     │
│     ├─► Is it EK1–EK4 (kill criterion)?
│     │     └─► Kill the direction. Document. Do not retry.
│     │
│     └─► Otherwise:
│           └─► Iterate once, within the budget (Section 7.5).
│
├─► Is the failure statistical? (Section 8)
│     │
│     ├─► Is it SK1 (insufficient sample)?
│     │     └─► Continue as exploratory. Adjust claims.
│     │
│     └─► Otherwise:
│           └─► Report honestly. Do not claim significance.
│
└─► Is the failure scope-related? (Section 9)
      └─► Reject the change or restart evaluation.
```

---

## 12. Escalation and Documentation

### 12.1 Every stop/pivot/kill decision is documented

A decision to stop, pivot, or kill is recorded in a decision log with:

- Date
- Triggering criterion (ID)
- Evidence (metric, check, artifact)
- Action taken
- Rationale
- Expected effect on the report

### 12.2 No silent overrides

An override of a criterion (allowing the project to continue despite a failure) requires:

1. A written justification.
2. Explicit labeling of subsequent results as **exploratory**.
3. A note in the final report.
4. An update to this document with a version bump.

### 12.3 Reporting post-decision

The final report includes:

- Which criteria fired
- What actions were taken
- What was **not** claimed as a result of criteria firing
- Any overrides and their justifications

---

## 13. Specific Kill Criteria for Each Direction

The following are the kill criteria for each specific direction in the project. They are pre-committed.

### 13.1 Kill the regime hypothesis (K > 1)

**Trigger:** After MVP + 2 iterations, `K > 1` never improves out-of-sample expected utility over `K = 1` by more than the bootstrap CI.

**Action:**
- Stop regime-aware modeling.
- Report the negative result.
- Release the benchmark as an honest negative finding.

### 13.2 Kill the decision-focused approach

**Trigger:** After MVP + 2 iterations, the decision-focused allocation does not beat 1/N or equal-weight in expected utility.

**Action:**
- Revert to baseline reporting.
- Report that decision-focused optimization did not add value in this setup.

### 13.3 Kill the macro feature hypothesis

**Trigger:** After MVP + 2 iterations, the model with macro features does not beat the model without them.

**Action:**
- Drop macro features from the headline model.
- Report the negative finding.

### 13.4 Kill the MoE direction

**Trigger:** After MVP + 2 iterations, no MoE variant beats a simple ridge regression under the same decision rule.

**Action:**
- Reframe the project around simpler models.
- Report MoE as an investigated-but-not-superior approach.

### 13.5 Kill the project

**Trigger:** Any of the following:
- T1, T2, T3 repeatedly fire with no fix.
- Data unavailable.
- Methodological criteria unsatisfiable.
- Phase 0 documentation cannot be made internally consistent.

**Action:**
- Archive the repository under tag `v2.0-abandoned`.
- Write a post-mortem documenting what was learned.
- Do not release claims.

---

## 14. What "Done" Looks Like

The project is **done** when:

1. The pre-code documentation is frozen.
2. The MVP pipeline runs end-to-end on default configuration.
3. Baselines and at least one regime-aware model are evaluated.
4. Primary metrics are reported with confidence intervals and multiple-testing corrections.
5. Robustness is reported across the declared dimensions.
6. The report complies with `07`, Section 12.
7. All criteria in this document have been applied.
8. Results are reproducible from configuration + raw data.
9. The public release includes the code, docs, and an explicit statement of limitations.

The project is **not** done if any of the above are missing, regardless of performance.

---

## 15. Ties to Other Documents

| Concern | Where specified |
|---------|-----------------|
| Invariants | `04`, Section 10 |
| Validation checks | `06`, Section 9 |
| Evaluation layers | `07`, Section 4 |
| Reporting standard | `07`, Section 12 |
| MVP architecture | `08`, Sections 6–10 |
| Contract and falsification tests | `08`, Section 17 |
| Scope boundaries | `03` |
| Research questions | `02`, Section 7 |

Conflicts are resolved by updating the affected document, not by silent override.

---

## 16. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Success is a rigorous, reproducible benchmark, not a high Sharpe. | Prevents overclaiming. |
| D2 | Criteria are binary, measurable, and pre-committed. | Prevents goalpost moving. |
| D3 | Relaxing a criterion downgrades results to exploratory. | Preserves honesty. |
| D4 | Iteration budget is 3 total (MVP + 2). | Prevents endless refinement. |
| D5 | Phase gates are binding. | Prevents proceeding without prerequisites. |
| D6 | Kill criteria exist for every direction. | Prevents indefinite pursuit. |
| D7 | Decisions are logged and reported. | Auditable. |
| D8 | A negative result is a valid outcome. | The benchmark's value is its rigor. |
| D9 | "Stop" and "kill" are distinct. | Preserves ability to pivot. |
| D10 | Overrides require downgrade to exploratory. | Prevents silent validity erosion. |

---

## 17. Open Questions

1. Should the iteration budget be 3 or 4 total? Current default: 3.
2. Should the kill criteria for the regime hypothesis use expected utility as the sole metric, or should there be a secondary metric check?
3. Is a "null model" (random weights) required for E1, or is 1/N sufficient as the floor?
4. Should E4 use 4 out of 9 robustness dimensions, or a different fraction?
5. Should Phase 3 (Primary Evaluation) require any positive result, or is a clean negative result also a valid exit?
6. Is the MVP phase gate too strict? A working pipeline that produces no decision value is still a valid exit if criteria fire at Phase 3.
7. Should kill criteria for individual models (e.g., MoE underperforms ridge) be checked at every iteration or only at the budget end?
8. Is `v2.0-abandoned` the correct tag, or should abandonment not be tagged at all?
9. Should statistical criteria apply to Layer 1 (predictive) metrics as strictly as to Layer 3 (decision) metrics?
10. What is the correct response when multiple criteria fire simultaneously (e.g., E1 and SK1)?
11. Should the decision log be a separate document (`docs/10_decision_log.md`), or embedded in `TODO.md`?
12. Is the current criterion set too complex for the MVP phase? Could a smaller subset suffice until Phase 3?
13. How should the criteria be tested during development? Should there be synthetic failures to validate that each criterion fires correctly?
14. Should statistical criteria be evaluated once per phase or continuously during evaluation?
15. Is the "two redirects" kill criterion in SCK3 too strict or too loose?

---

## 18. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial stop criteria and kill criteria for v2. |

---

## 19. References

- `00_docs_index.md` — documentation map and conventions.
- `01_first_principles_problem_decomposition.md` — primitive decomposition.
- `02_problem_framing.md` — formal problem statement and research questions.
- `03_scope_and_non_goals.md` — scope boundaries and non-goals.
- `04_assumptions_and_invariants.md` — assumptions and invariants.
- `05_objective_utility_and_decision_specification.md` — objective, utility, decision rule.
- `06_data_information_and_target_contract.md` — data contract.
- `07_evaluation_framework_and_backtesting_protocol.md` — evaluation protocol.
- `08_minimum_viable_baseline_end_to_end_pipeline_architecture.md` — MVP architecture.
- v1 `TODO.md` — archived under `archive/v1/TODO.md`, for the pattern of stopping decisions in practice.
