# Docs Index

**Project:** `moe-factor-timing` — v2 (Decision-Focused Regime-Aware Factor Allocation)
**Folder:** `docs/`
**Status:** Pre-code documentation set
**Last Updated:** 2026-08-03

---

## 0. Purpose of This Index

This file is the entry point to the v2 documentation.

The documents in this folder are written **before any v2 code is written**. Their purpose is to force clarity on:

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
| 01 | `01_first_principles_problem_decomposition.md` | Break the problem down to primitives. Question inherited assumptions. Identify the root problem. | ⬜ Not started |
| 02 | `02_problem_framing.md` | Convert the decomposition into a formal problem statement, research questions, and formal specification. | ⬜ Not started |
| 03 | `03_scope_and_non_goals.md` | Explicit boundaries: what is in, what is out, and what this project will not claim. | ⬜ Not started |
| 04 | `04_assumptions_and_invariants.md` | Every assumption stated, challenged, and either accepted, rejected, or marked as a hypothesis to test. Invariants that must never be violated. | ⬜ Not started |
| 05 | `05_objective_utility_and_decision_specification.md` | The investor objective, utility function, decision rule, and constraint set. Prediction is not the goal; decisions are. | ⬜ Not started |
| 06 | `06_data_information_and_target_contract.md` | Data sources, vintage policy, information set at time `t`, feature contract, target variable, prediction horizon, no-look-ahead rules. | ⬜ Not started |
| 07 | `07_evaluation_framework_and_backtesting_protocol.md` | Predictive, decision, and statistical metrics. Walk-forward protocol. Cost model. Multiple-testing discipline. Reporting standards. | ⬜ Not started |
| 08 | `08_minimum_viable_baseline_end_to_end_pipeline_architecture.md` | The smallest end-to-end system that can produce a valid, reproducible result. Module boundaries, data flow, interfaces. | ⬜ Not started |
| 09 | `09_stop_criteria_and_kill_criteria.md` | Pre-committed conditions for stopping, pivoting, or killing the project. Prevents goalpost-moving after seeing results. | ⬜ Not started |

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

- `03` and `04` can be written in parallel once `02` is stable.
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

---

## 5. Document Conventions

Every document in this folder must follow the same structure:

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
5. **No code before `09`.** Implementation begins only after the pre-code set is complete and consistent.

---

## 6. What This Project Is Not

This documentation set exists partly to prevent the failure modes of v1. Explicitly:

- This is not a "beat the market" project.
- This is not a "prove MoE works" project.
- This is not a prediction accuracy competition.
- This is not a demonstration that a backtest Sharpe ratio implies live performance.
- This is not a project that treats 42 out-of-sample months as statistical evidence.

These non-claims are expanded in `03_scope_and_non_goals.md`.

---

## 7. Relationship to v1

v1 is treated as a **reference implementation and benchmark**, not as the problem definition.

What v1 contributes:

- A working data pipeline and backtest harness.
- A baseline model suite (persistence, rolling average, momentum, linear, random forest, SimpleMoE).
- A transaction cost model.
- Evidence that predictive accuracy and allocation quality can diverge.

What v1 does not settle:

- Whether the problem was framed correctly.
- Whether the information set was valid.
- Whether the evaluation was statistically meaningful.
- Whether the allocation rule was derived from an objective.

v2 starts from `01_first_principles_problem_decomposition.md` and rebuilds the problem definition independently of v1's implementation choices.

---

## 8. Open Questions

1. Which documents in the pre-code set should be frozen before implementation, versus allowed to remain in `Draft`?
2. Is `09_stop_criteria_and_kill_criteria.md` written before or after `08_minimum_viable_baseline_end_to_end_pipeline_architecture.md`? Current order assumes after.
3. Should the v1 repository be archived as a tag or kept as a separate branch for v2 comparison?
4. What is the single most important thing that must be true for v2 to be worth building?

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-08-03 | Initial index for the v2 pre-code documentation set. |

---

## 10. References

- `README.md` — v1 project overview.
- `problem_framing.md` — v1 problem framing (reference only).
- `TODO.md` — v1 build log and bug tracker (reference only).
