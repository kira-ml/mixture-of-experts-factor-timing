# moe-factor-timing v2

**Decision-Focused Regime-Aware Factor Allocation**

A reproducible benchmark for evaluating whether probabilistic
regime-aware models improve out-of-sample factor allocation decisions
under realistic costs and constraints.

---

## Result

Under the tested configurations, **no probabilistic model beats the
1/N benchmark on expected utility after costs**. The result is robust
across training-window length and number of latent states. It survives
a fully validated pipeline (determinism, solver fallback, leakage
detection).

**Secondary finding.** A calibrated predictive distribution does not
imply decision value: a two-state Gaussian HMM that passes PIT
calibration in one configuration still trails 1/N by 45% on expected
utility.

- Full report: [`results/v2_report.md`](results/v2_report.md)
- Experiment log: [`docs/10_experiment_log.md`](docs/10_experiment_log.md)
- Phase 1 validations: [`docs/11_phase1_validation_log.md`](docs/11_phase1_validation_log.md)

---

## What this is

- A decision-focused evaluation framework for factor allocation.
- A pre-committed protocol (`docs/07`, `docs/09`) that separates model
  selection from evaluation.
- A reproducible pipeline where every result traces to a configuration
  and a raw data snapshot.
- A negative result, rigorously established.

## What this is not

- Not a trading strategy.
- Not a fund.
- Not investment advice.
- Does not claim to beat any benchmark.
- Does not claim regimes exist in the market.
- Does not claim backtest results imply live performance.
- Does not claim statistical significance without tests.

See `docs/03_scope_and_non_goals.md` Section 9 for the full list.

---

## Repository layout

```
├── configs/            Run configurations (YAML)
├── docs/               Pre-code documentation (00–11)
├── results/            Curated, committed outputs
├── runs/               Immutable run outputs (git-ignored)
├── src/                Ten-stage pipeline
├── tests/              Contract and falsification tests
├── scripts/            Analysis scripts (diagnostics, sweeps)
├── archive/v1/         v1 implementation (tag v1.0-final)
└── pyproject.toml
```

## Pipeline

Ten stages, each with its own contract (`docs/08` Section 7):

```
ingest → validate → features → split → models → decision → backtest
       → evaluate → report → run
```

## Reproduce the reference run

```bash
pip install -e ".[dev]"
echo "FRED_API_KEY=your_key_here" > .env
python -m src.run --config configs/default.yaml
```

Determinism check:

```bash
python scripts/check_determinism.py
```

Contract and falsification tests:

```bash
pytest tests/ -v
```

---

## Key documentation

| File | Purpose |
|---|---|
| `docs/02_problem_framing.md` | Formal problem statement and research questions |
| `docs/05_objective_utility_and_decision_specification.md` | Objective, utility, decision rule |
| `docs/07_evaluation_framework_and_backtesting_protocol.md` | Evaluation protocol |
| `docs/09_stop_criteria_and_kill_criteria.md` | Pre-committed stop criteria |
| `docs/10_experiment_log.md` | Data-driven reasoning trail through Phase 5 |
| `docs/11_phase1_validation_log.md` | Six pipeline validations |
| `results/v2_report.md` | Final report |

---

## Limitations

- 49-month out-of-sample window; underpowered for Sharpe claims.
- Mean-variance utility only; CRRA and CVaR not evaluated.
- Assumed 10 bps costs; calibrated costs not tested.
- FRED latest + 1-month lag substituted for ALFRED vintages.
- Single asset class (US equity factors), single frequency (monthly).

See `results/v2_report.md` Section 11 for the full list.

---

## License

MIT
