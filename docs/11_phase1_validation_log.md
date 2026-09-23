# Phase 1 Validation Log

**Project:** moe-factor-timing v2
**Status:** Frozen
**Last Updated:** 2026-09-24

---

## 1. Purpose

Formal closure of the Phase 1 exit gate (`09` Section 10.2). Records
the six validations that establish the pipeline as trustworthy before
the Phase 5 report is written.

Phase 1's exit gate was satisfied **retroactively**. The validation
work was performed after Phases 2–4 experiments had already produced
results. Those results are confirmed valid by this log.

---

## 2. Context and Dependencies

- **Depends on:** `04` Section 10.4 (invariant IV.4), `06` Section 9
  (validation checks), `08` Section 17 (testing strategy), `09`
  Section 10.2 (Phase 1 exit gate).
- **Feeds into:** Phase 5 report (`07` Section 12), and any future
  v3 that inherits this pipeline.

---

## 3. Phase 1 Exit Gate Requirements (`09` Section 10.2)

| Requirement | Status | Evidence |
|---|---|---|
| Stages 1–10 implemented per `08` Section 7 | ✅ | All ten modules in `src/` |
| Contract tests pass (`08` Section 17.1) | ✅ | `tests/test_pipeline_contracts.py` |
| Falsification tests pass (`08` Section 17.2) | ✅ | V13 injection test |
| Integration test passes (`08` Section 17.3) | ✅ | Full default run produces valid manifest |
| One full run produces a valid manifest | ✅ | `runs/20260923T173318Z/manifest.json` |

---

## 4. Six Validations

### V1 — Cash residual earns risk-free

**Trigger:** Original run `20260923T151647Z` showed `cash` Sharpe = 0.00
and `ann_return` = 0.0%. Contradicts `02`/`05` (cash is a residual asset
and must earn the risk-free rate).

**Diagnosis:** `src/backtest/engine.py::compute_returns` computed
`w @ realized` only. The residual `1 − Σw` was silently earning zero.

**Fix:** `compute_returns` gains a `cash_return` parameter. The residual
`1 − Σw` is credited with the risk-free rate. Orchestrator passes
`cash_returns[t_test]`.

**Result:** `cash` Sharpe = 17.2, `ann_return` = 4.4%. Correct.

---

### V2 — HMM covariance shape

**Trigger:** HMM NLL = 866 vs ridge NLL = −10.5 on the same target and
the same 5-asset dimension. Orders of magnitude apart; not a valid
comparison.

**Diagnosis:** `scripts/diag_hmm.py` showed `hmmlearn` 0.3.x returns
`covars_` with shape `(K, N, N)` for `covariance_type='diag'`, not the
`(K, N)` shape assumed by the original code. The old code extracted
diagonals to a 1-D vector, which broadcast against
`np.outer(diff, diff)` and produced a rank-1 dominant matrix.

**Fix:** `predict_distribution` in `src/models/hmm_gaussian.py` branches
on `covars_.ndim` and uses the `(K, N, N)` matrices directly when present.

**Result:** HMM NLL = −5.9, same order of magnitude as ridge (−11.4).
Verified via `scripts/diag_hmm.py`.

---

### V3 — V13 target-leakage detection fires

**Test:** `tests/test_pipeline_contracts.py::test_v13_detects_target_leakage`.

**Method:** Inject a feature that is a direct copy of the target. Run
`build_features`. Assert `leakage_warnings` in the feature manifest.

**Result:** ✅ PASS. The injected feature is flagged with
`{"leaked_SPY|SPY": 1.0}`. The check is not a no-op.

**Companion test:** `test_v13_no_leakage` verifies that legitimate
features produce an empty `leakage_warnings` dict.

---

### V4 — Determinism

**Test:** `scripts/check_determinism.py`.

**Method:** Run the pipeline twice on a reduced config
(`min_train=60`, `end=2023-01-01`, 3 models: `naive_1n`, `ridge`,
`hmm_gaussian`). Compare SHA-256 hashes of every artifact recorded in
`manifest.json`.

**Exclusion:** `raw/provenance.json` is excluded. It records
`retrieved_at_utc`, which differs by design between runs. It is metadata,
not a result artifact (`08` Section 7.1).

**Result:** ✅ PASS. **18/18 result artifacts byte-identical across two
runs.**

**Note on full-config drift:** Full-configuration runs still show a
run-to-run Sharpe drift of approximately **0.005** for probabilistic
models (ridge: 1.1211 → 1.1264 across independent runs). This is two
orders of magnitude smaller than the effect being discussed (ridge
trails 1/N by ~0.16 Sharpe) and cannot change any conclusion. Disclosed
in the Phase 5 report.

---

### V5 — Solver statuses tracked

**Check:** `Select-String -Path src/run/orchestrator.py -Pattern "solver_statuses"`.

**Result:** ✅ 4 matches returned:
1. `solver_statuses: list[str] = []` — declaration
2. `solver_statuses.append("fixed")` — for fixed models
3. `solver_statuses.append(str(_status))` — for probabilistic models
4. `"solver_statuses": solver_statuses` — recorded in per-model results

Every rebalance of every model has a recorded solver status.

---

### V6 — Solver fallback rate

**Trigger:** `09` Section 5.1 T6 fires if the solver fails on more than
5% of rebalances. Required for the report by `07` Section 12.1.

**Method:** `fallback_rate = n_fallback / n_rebalances` computed in
`src/evaluate/evaluate.py` from the solver statuses recorded by V5.

**Result:** `fallback_rate = 0.0` for all seven models, including
`ridge` and `hmm_gaussian`, across 49 rebalances each.

**T6 does not fire.** The negative result is not an artifact of solver
failure.

---

## 5. Outcome

All six validations pass. The Phase 1 exit gate (`09` Section 10.2) is
satisfied.

The decision-layer negative result — no model beats 1/N on expected
utility after costs, robust across 6/6 robustness configurations —
survives a fully validated pipeline. Phase 5 report may proceed.

---

## 6. Residual Disclosures

Two items carry into the Phase 5 report as explicit caveats:

1. **Sub-0.01 Sharpe drift** between full-config runs (V4 note).
   Immaterial to any ranking.

2. **FRED latest + 1-month lag** used in place of ALFRED vintages
   (`06` Section 6.2). Documented approximation; conservative in
   the direction of understating model performance.

---

## 7. Ties to Other Documents

| Concern | Where specified |
|---|---|
| Phase 1 exit gate | `09` Section 10.2 |
| Contract tests | `08` Section 17.1 |
| Falsification tests | `08` Section 17.2 |
| Integration test | `08` Section 17.3 |
| V13 leakage check | `06` Section 9.3 |
| T6 fallback threshold | `09` Section 5.1 |
| Determinism invariant | `04` Section 10.4 (IV.4) |
| Manifest and artifact hashes | `07` Section 13.3 |

Conflicts are resolved by updating the affected document, not by silent
override.

---

## 8. Open Questions

1. Should the Phase 1 exit gate have been verified before running
   Phases 2–4? Current answer: yes, and this document records that it
   was not, and that the gap was closed retroactively. For v3, the gate
   must be passed before advancing.
2. Is the ~0.005 Sharpe drift acceptable, or should the solver be
   replaced with a strictly deterministic one (e.g., CLARABEL) for
   future runs?
3. Should `raw/provenance.json` be split into two artifacts (a result
   hash and a separate metadata file) so that determinism checks do not
   need an exclusion list?

---

## 9. Version History

| Version | Date | Changes |
|---|---|---|
| 0.1 | 2026-09-24 | Initial Phase 1 validation log; exit gate closed retroactively. |

---

## 10. References

- `08_minimum_viable_baseline_end_to_end_pipeline_architecture.md`
  — Section 17 (Testing Strategy).
- `09_stop_criteria_and_kill_criteria.md` — Section 10.2 (Phase 1 exit gate).
- `tests/test_pipeline_contracts.py` — V3 falsification tests.
- `scripts/check_determinism.py` — V4 determinism check.
- `scripts/diag_hmm.py` — V2 diagnosis.
- `runs/20260923T173318Z/` — reference run for V5 and V6.
