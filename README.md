# moe-factor-timing

## Mixture of Experts for Regime-Switching Factor Timing

🚧 **Week 1 MVP** — Data pipeline and baseline models

---

## The Problem

Equity factor premiums (Value, Momentum, Quality, Low Volatility) are notoriously unstable. Value can underperform for a decade before roaring back. Momentum can crash suddenly. This creates a critical problem for investors:

**How do you allocate to factors when you don't know which regime will dominate next?**

Most approaches fall short:
- **Ignore regimes** → Static allocation suffers prolonged drawdowns
- **Binary regime labels** → Overconfident bets that fail spectacularly
- **Short-term momentum** → Noisy and unreliable signals

---

## The Research Question

> Can a probabilistic model that represents uncertainty over economic regimes improve factor timing compared to simpler deterministic approaches?

---

## Our Approach

We frame this as a **probabilistic time-series forecasting** problem:

**Inputs:**
- Factor returns (Value, Momentum, Quality, Low Volatility, Size)
- Macroeconomic indicators (inflation, production, unemployment, term spread)

**Output:**
- Probability distribution of next-month factor returns
- Each mixture component = a latent economic regime

**Evaluation:**
- Predictive accuracy (log-likelihood)
- Investment performance (Sharpe ratio of dynamic allocation)

---

## Models

| Model | Description |
|-------|-------------|
| Persistence | Next month = current month (naïve baseline) |
| Rolling Average | Next month = average of past N months |
| Linear Regression | Predict from lagged features + macro |
| Random Forest | Non-linear ensemble |
| HMM + Regression | Two-stage regime identification |
| **Mixture of Experts** | End-to-end LSTM gating + regime experts |

---

## Data Sources (MVP)

- **yfinance** — Factor proxies (SPY, IWD, MTUM, QUAL, USMV)
- **yfinance** — VIX (macro proxy)

*Later: French Data Library + FRED for academic-grade factors.*

---

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/moe-factor-timing.git
cd moe-factor-timing

# Install
pip install -r requirements.txt

# Download data
python -c "from src.data_pipeline import DataPipeline; DataPipeline().align_data()"

# Explore
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # Downloaded data
│   └── processed/    # Cleaned/aligned data
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── models.py
│   └── evaluation.py
├── tests/
└── results/
    └── figures/
```

---

## Next Steps

- [x] Project structure
- [ ] Data pipeline with yfinance
- [ ] Baseline models
- [ ] Backtesting framework
- [ ] HMM implementation
- [ ] Mixture of Experts (PyTorch)
- [ ] Final evaluation and results

---

## License

MIT
```

---

## Alternative: Shorter Version

```markdown
# moe-factor-timing

## Mixture of Experts for Regime-Switching Factor Timing

🚧 Week 1 MVP

---

### The Problem

Factor premiums are time-varying and regime-dependent. Most approaches use simple momentum or binary regime labels. We test whether probabilistic regime identification improves factor timing.

**Research Question:** Can explicit uncertainty over economic regimes improve out-of-sample factor timing?

---

### Approach

Probabilistic time-series forecasting with latent regimes.

**Inputs:** Factor returns + macro indicators
**Outputs:** Distribution of next-month factor returns
**Evaluation:** Log-likelihood + Sharpe ratio

---

### Models

| Type | Models |
|------|--------|
| Baselines | Persistence, Rolling Average, Linear Regression, Random Forest |
| Regime | HMM + Regression |
| Advanced | Mixture of Experts (LSTM gating) |

---

### Quick Start

```bash
pip install -r requirements.txt
python src/data_pipeline.py
```

### Data

yfinance for MVP (SPY, IWD, MTUM, QUAL, USMV).

### Next Steps

- [ ] Data pipeline
- [ ] Baselines
- [ ] Backtesting
- [ ] MoE implementation

### License

MIT