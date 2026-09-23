# Docs Index

**Project:** `moe-factor-timing` — v2 (Decision-Focused Regime-Aware Factor Allocation)
**Folder:** `docs/`
**Status:** Pre-code documentation set — Drafted
**Last Updated:** 2026-08-03

---

## 0. Purpose of This Index

This file is the entry point to the v2 documentation.

The documents in this folder were written **before any v2 code was written**. Their purpose is to force clarity on:

1. What problem is actually being solved.
2. Why it is being solved that way.
3. What counts as valid evidence.
4. What the minimum viable system must do.
5. When to stop, pivot, or kill the project.

If a design decision cannot be traced back to one of these documents, it should not be implemented yet.

---

## 1. How to Use This Folder

- Read documents in numeric order. Later documents assume earlier ones.
- Do not write v2 code until documents `01` through `09` exist and are internally consistent.
- Each document must contain a **Version History** section.
- Each document must end with an **Open Questions** section. Unresolved questions are allowed; hidden assumptions are not.
- If a change in an earlier document invalidates a later one, update the later one in the same commit.

---

## 2. Pre-Code Documentation Set

These are the documents that must be established before coding begins.

| # | File | Purpose | Status |
|---|------|---------|--------|
| 00 | `00_docs_index.md` | Entry point, conventions, document map. | ✅ Drafted |
| 01 | `01_first_principles_problem_decomposition.md` | Break the problem down to primitives. Question inherited assumptions. Identify the root problem. | ✅ Drafted |
| 02 | `02_problem_framing.md` | Convert the decomposition into a formal problem statement, research questions, and formal specification. | ✅ Drafted |
| 03 | `03_scope_and_non_goals.md` | Explicit boundaries: what is in, what is out, and what this project will not claim. | ✅ Drafted |
| 04 | `04_assumptions_and_invariants.md` | Every assumption stated, challenged, and either accepted, rejected, or marked as a hypothesis to test. Invariants that must never be violated. | ✅ Drafted |
| 05 | `05_objective_utility_and_decision_specification.md` | The investor objective, utility function, decision rule, and constraint set. Prediction is not the goal; decisions are. | ✅ Drafted |
| 06 | `06_data_information_and_target_contract.md` | Data sources, vintage policy, information set at time `t`, feature contract, target variable, prediction horizon, no-look-ahead rules. | ✅ Drafted |
| 07 | `07_evaluation_framework_and_backtesting_protocol.md` | Predictive, decision, and statistical metrics. Walk-forward protocol. Cost model. Multiple-testing discipline. Reporting standards. | ✅ Drafted |
| 08 | `08_minimum_viable_baseline_end_to_end_pipeline_architecture.md` | The smallest end-to-end system that can produce a valid, reproducible result. Module boundaries, data flow, interfaces. | ✅ Drafted |
| 09 | `09_stop_criteria_and_kill_criteria.md` | Pre-committed conditions for stopping, pivoting, or killing the project. Prevents goalpost-moving after seeing results. | ✅ Drafted |

**All pre-code documents are drafted.** The remaining step before implementation is to review each document, resolve open questions where possible, and freeze the set.

---

## 3. Reading Order and Dependencies

```text
00_docs_index.md
   │
   ▼
01_first_principles_problem_decomposition.md
   │
   ▼
02_problem_framing.md
   │
   ├──▶ 03_scope_and_non_goals.md
   │
   ├──▶ 04_assumptions_and_invariants.md
   │
   ▼
05_objective_utility_and_decision_specification.md
   │
   ▼
06_data_information_and_target_contract.md
   │
   ▼
07_evaluation_framework_and_backtesting_protocol.md
   │
   ▼
08_minimum_viable_baseline_end_to_end_pipeline_architecture.md
   │
   ▼
09_stop_criteria_and_kill_criteria.md
```

Notes:

- `03` and `04` can be reviewed in parallel once `02` is stable.
- `05` depends on `02` and `04`. It defines what "good" means before any metric is chosen.
- `06` depends on `05`. The decision problem determines what data is required, not the other way around.
- `07` depends on `05` and `06`. Evaluation must match the decision and the information set.
- `08` depends on `06` and `07`. Architecture serves the contract and the evaluation.
- `09` depends on `02`, `05`, and `07`. Stop criteria must reference the problem, the objective, and the evaluation.

---

## 4. Deferred Documentation (Post-MVP)

The following documents are intentionally **out of scope for the pre-code phase**. They will be created while building the system, not before. They are listed here so they are not forgotten.

| File | Reason for Deferral |
|------|---------------------|
| `10_model_hypotheses_and_model_zoo.md` | Depends on empirical behavior of the MVP. |
| `11_regime_definitions_and_identification.md` | Depends on data contract and MVP results. |
| `12_transaction_costs_capacity_and_constraints.md` | Requires cost calibration from real instruments. |
| `13_statistical_validation_and_multiple_testing.md` | Requires the actual experiment grid. |
| `14_reproducibility_and_experiment_tracking.md` | Requires the chosen tooling. |
| `15_risk_management_and_failure_modes.md` | Requires the live pipeline to be observable. |
| `16_results_reporting_and_visualization_standards.md` | Requires first real results. |
| `17_data_dictionary.md` | Requires the finalized feature set. |
| `18_api_contracts_and_interfaces.md` | Requires stable module boundaries. |
| `19_testing_and_validation_plan.md` | Requires the implementation to test. |
| `20_roadmap_and_milestones.md` | Requires an estimate of remaining work. |
| `21_decision_log_and_version_history.md` | Grows continuously during the build. |
| `22_glossary.md` | Grows continuously during the build. |
| `23_references.md` | Grows continuously during the build. |

Do not create these prematurely. Premature documentation of unknowns becomes fiction.

**Exception:** `21_decision_log_and_version_history.md` is created at Phase 0 exit, because stop/pivot/kill decisions in `09` require a place to be recorded.

---

## 5. Document Conventions

Every document in this folder follows the same structure:

```markdown
# <Title>

**Project:** moe-factor-timing v2
**Status:** Draft | Review | Frozen
**Last Updated:** YYYY-MM-DD

---

## 1. Purpose
## 2. Context and Dependencies
## 3. Body
## 4. Decisions
## 5. Open Questions
## 6. Version History
## 7. References
```

Rules:

1. **Status discipline.**
   - `Draft` — actively being written, may change.
   - `Review` — content complete, awaiting critique.
   - `Frozen` — code may depend on it. Changes require a version bump.
2. **No hidden assumptions.** Every assumption is either stated in `04_assumptions_and_invariants.md` or explicitly rejected.
3. **No goals without a metric.** Any stated objective must reference a measurable quantity in `07_evaluation_framework_and_backtesting_protocol.md`.
4. **No metric without a decision.** Any metric must be traceable to the decision specification in `05_objective_utility_and_decision_specification.md`.
5. **No code before `09` is frozen.** Implementation begins only after the pre-code set is complete and consistent.

---

## 6. What This Project Is Not

This documentation set exists partly to prevent the failure modes of v1. Explicitly:

- This is not a "beat the market" project.
- This is not a "prove MoE works" project.
- This is not a prediction accuracy competition.
- This is not a demonstration that a backtest Sharpe ratio implies live performance.
- This is not a project that treats 42 out-of-sample months as statistical evidence.

These non-claims are expanded in `03_scope_and_non_goals.md` and must appear verbatim in the final report (`03`, Section 9).

---

## 7. Relationship to v1

v1 is **archived** and treated as a reference implementation and benchmark, not as the problem definition.

### 7.1 Archive location

```
archive/v1/
├── src/                  # v1 source modules
├── main.py               # v1 orchestrator
├── run_moe_torch.py      # v1 PyTorch experiment
├── data/                 # v1 raw + processed data (git-ignored)
├── notebooks/            # v1 notebooks
├── scripts/              # v1 utility scripts
├── logs/                 # v1 run logs
├── requirements_v1.txt   # v1 dependencies
├── README_v1.md          # v1 README
├── problem_framing.md    # v1 problem framing
└── TODO.md               # v1 build log
```

### 7.2 v1 recovery

- v1 is recoverable via the git tag `v1.0-final`.
- v1 code is **not** on the v2 import path.
- v1 data is **not** used by v2 runs.

### 7.3 What v1 contributes

- A working data pipeline and backtest harness.
- A baseline model suite (persistence, rolling average, momentum, linear, random forest, SimpleMoE).
- A transaction cost model.
- Evidence that predictive accuracy and allocation quality can diverge.

### 7.4 What v1 does not settle

- Whether the problem was framed correctly.
- Whether the information set was valid.
- Whether the evaluation was statistically meaningful.
- Whether the allocation rule was derived from an objective.

### 7.5 What v2 replaces

| v1 choice | v2 replacement |
|-----------|----------------|
| Flat `src/` layout | Ten-stage modular layout (`08`, Section 7) |
| Ad hoc allocation rules | Decision rule derived from objective (`05`) |
| Single-shot Sharpe reporting | Layered evaluation with CIs (`07`) |
| Implicit model selection | Explicit separation of selection and evaluation (`07`, Section 10.6) |
| Outputs inside `src/results/` | Immutable `runs/<run_id>/` + curated `results/` |
| Assumed data availability | Vintage-aware contract (`06`) |
| Look-ahead risks in features | Enforced validation checks (`06`, Section 9) |

v2 starts from `01_first_principles_problem_decomposition.md` and rebuilds the problem definition independently of v1's implementation choices.

---

## 8. Project Lifecycle

The project proceeds through six phases, defined in `09`, Section 10.

| Phase | Name | Entry | Exit |
|-------|------|-------|------|
| 0 | Documentation | Project initiated | `docs/00`–`docs/09` frozen, v1 archived |
| 1 | Pipeline MVP | Phase 0 exit | Full pipeline runs end-to-end, contract tests pass |
| 2 | Baseline Evaluation | Phase 1 exit | Baselines + one regime-aware model evaluated |
| 3 | Primary Evaluation | Phase 2 exit | Empirical and statistical criteria applied |
| 4 | Robustness | Phase 3 exit | Robustness across declared dimensions |
| 5 | Report | Phase 4 exit | Report compliant with `07`, Section 12 |
| 6 | Release | Phase 5 exit | Repository tagged `v2.0` |

**Current phase:** Phase 0 — Documentation.
**Next action:** Review and freeze `docs/00`–`docs/09`.

---

## 9. Open Questions

1. Which documents in the pre-code set should be frozen before implementation, versus allowed to remain in `Draft`?
2. Should `docs/10_decision_log_and_version_history.md` be created at Phase 0 exit, or deferred until the first stop/pivot/kill decision is made?
3. Should the `docs/00`–`docs/09` set be frozen together, or frozen incrementally as each is reviewed?
4. What is the single most important thing that must be true for v2 to be worth building?
5. Should Phase 0's exit gate require a formal review of each document, or is a self-review sufficient?
6. Should `pyproject.toml` and `requirements.txt` be written before Phase 1, or during Phase 1?
7. Should the first v2 commit after freezing the docs be a tag (e.g., `v2.0-docs-frozen`)?

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial index for the v2 pre-code documentation set. |
| 0.2 | 2026-08-03 | Updated status of `01`–`09` to Drafted. Added v1 archive contract, project lifecycle table, and updated open questions. |

---

## 11. References

- `docs/01`–`docs/09` — pre-code documentation set.
- `archive/v1/README_v1.md` — v1 project overview (reference only).
- `archive/v1/problem_framing.md` — v1 problem framing (reference only).
- `archive/v1/TODO.md` — v1 build log (reference only).
- Git tag `v1.0-final` — exact v1 recovery point.
