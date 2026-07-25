# TODO.md

## Mixture of Experts for Regime-Switching Factor Timing

**Project:** `moe-factor-timing`  
**Repository:** https://github.com/kira-ml/mixture-of-experts-factor-timing.git  
**Start Date:** July 25, 2026  
**Current Status:** Week 1-2 MVP Complete

---

## 📅 DAILY LOG

### July 25, 2026 (Day 1) - Project Initiation & Week 1 MVP

#### Completed Tasks

**Morning Session - Project Setup**
- [x] Created project repository structure
- [x] Set up virtual environment (`venv`)
- [x] Created `requirements.txt` with core dependencies
- [x] Wrote initial `README.md` with project overview
- [x] Created `problem_framing.md` with research question and methodology
- [x] Initialized Git repository and pushed to GitHub

**Afternoon Session - Data Pipeline**
- [x] Implemented `src/data_pipeline.py` with yfinance integration
- [x] Added support for factor proxies (SPY, IWD, MTUM, QUAL, USMV, VIX)
- [x] Implemented multi-index column handling for yfinance data
- [x] Added monthly resampling and return calculation
- [x] Created data validation and sanity checks
- [x] Saved processed data to `data/processed/`
- [x] Fixed column detection issues after resampling

**Evening Session - Baseline Models**
- [x] Implemented `src/models.py` with 5 baseline models:
  - [x] Persistence Model (next month = current month)
  - [x] Rolling Average Model (average of last N months)
  - [x] Momentum Model (exponentially weighted historical returns)
  - [x] Linear Regression (scikit-learn)
  - [x] Random Forest (scikit-learn)
- [x] Added multi-output support for all models (6 factors)
- [x] Created factory function `create_model()` for easy instantiation
- [x] Added consistent `fit()`/`predict()` API across all models

**Late Session - Evaluation & Backtesting**
- [x] Implemented `src/evaluation.py` with metrics:
  - [x] RMSE, MAE (predictive metrics)
  - [x] Sharpe ratio (annualized)
  - [x] Maximum drawdown
  - [x] Annualized return
  - [x] Volatility
  - [x] Calmar ratio
  - [x] Win rate
- [x] Implemented `src/backtest.py` with expanding window backtest
- [x] Added `tqdm` progress bar for backtest monitoring
- [x] Implemented `summarize_results()` and `get_best_model()` functions
- [x] Added `src/utils.py` with shared helpers:
  - [x] Date handling utilities
  - [x] Data validation utilities
  - [x] Configuration helpers
  - [x] Logging setup
  - [x] Random seed management

**Late Night - Orchestration & MoE**
- [x] Created `main.py` as main orchestrator
- [x] Added command-line argument parsing
- [x] Integrated data pipeline, backtest, and results display
- [x] Implemented `src/moe.py` - Simple Mixture of Experts:
  - [x] Linear experts (scikit-learn)
  - [x] Softmax gating (scipy)
  - [x] EM algorithm training
  - [x] Multi-output support
  - [x] Regime probability extraction
- [x] Added MoE to model registry
- [x] Ran full pipeline successfully

#### Current Results

| Model | RMSE | MAE | Sharpe Ratio |
|-------|------|-----|--------------|
| **MoE** | **11.7350** | **6.6553** | 1.7667 |
| Rolling Avg | 12.2288 | 6.7109 | 1.7667 |
| Momentum | 12.3836 | 6.7823 | 1.7667 |
| RF | 13.7637 | 7.3650 | 1.7667 |
| Linear | 14.9084 | 8.2372 | 1.7667 |
| Persistence | 17.8404 | 9.8470 | 1.7667 |

**Key Finding:** MoE outperforms all baselines with 4% improvement in RMSE over Rolling Average.

#### Challenges Encountered & Resolved

- [x] **Issue:** yfinance MultiIndex columns causing 'Close' key error
  - **Solution:** Added robust column detection for MultiIndex DataFrames

- [x] **Issue:** Models predicting single output instead of 6 factors
  - **Solution:** Added multi-output support to all models

- [x] **Issue:** Unicode emoji (✅) causing logging error in Windows
  - **Status:** Known issue, non-critical (logging still works)

- [x] **Issue:** Missing `tqdm` dependency
  - **Solution:** Added to requirements.txt

---

## 🎯 PROJECT TODO LIST

### Week 1: Data Pipeline & Baselines ✅ COMPLETED

- [x] Project structure setup
- [x] Data pipeline with yfinance
- [x] Baseline models (5 models)
- [x] Evaluation framework
- [x] Backtesting framework
- [x] Main orchestrator
- [x] Simple MoE implementation

### Week 2: Analysis & Visualization (July 26 - August 1, 2026)

#### Immediate Tasks

- [ ] **Fix Investment Metrics** (Critical)
  - Implement portfolio allocation based on predictions
  - Replace actual returns with model-based allocation returns
  - Calculate transaction-cost-adjusted returns
  - Compare Sharpe ratios of different allocation strategies

- [ ] **Analyze MoE Results**
  - [ ] Extract and plot regime probabilities over time
  - [ ] Map regimes to known economic events:
    - [ ] COVID-19 pandemic (2020)
    - [ ] 2022 bear market
    - [ ] Post-COVID recovery
  - [ ] Analyze expert specialization (what each expert learned)

- [ ] **Visualization Suite**
  - [ ] Cumulative returns comparison plot
  - [ ] Regime probability heatmap
  - [ ] Factor allocation time series
  - [ ] Performance metrics bar chart
  - [ ] Correlation matrix of predictions

- [ ] **Hyperparameter Tuning**
  - [ ] MoE: Try K=2, 3, 4, 5 experts
  - [ ] MoE: Test different EM iterations (30, 50, 100)
  - [ ] Rolling Avg: Test different windows (3, 6, 12, 24)
  - [ ] Momentum: Test different decay rates (0.7, 0.8, 0.9, 0.95)
  - [ ] RF: Grid search for n_estimators and max_depth

- [ ] **Statistical Testing**
  - [ ] Diebold-Mariano test for predictive accuracy
  - [ ] Bootstrap confidence intervals for Sharpe ratios
  - [ ] Effect size calculations

### Week 3: HMM Implementation (August 2-8, 2026)

- [ ] **Hidden Markov Model**
  - [ ] Implement HMM with `hmmlearn` library
  - [ ] Two-stage approach: identify regimes, then predict
  - [ ] Compare with MoE regime detection
  - [ ] Visualize HMM states vs MoE regimes

- [ ] **Feature Engineering**
  - [ ] Add FRED macro indicators (CPI, IP, UNRATE, term spread)
  - [ ] Create rolling volatility features
  - [ ] Add cross-sectional factor correlations

### Week 4: PyTorch MoE (August 9-15, 2026)

- [ ] **PyTorch Setup**
  - [ ] Uncomment PyTorch in requirements.txt
  - [ ] Install PyTorch with CUDA (if available)

- [ ] **Deep MoE Implementation**
  - [ ] LSTM gating network (processes sequences)
  - [ ] Neural network experts (non-linear)
  - [ ] Joint training with gradient descent
  - [ ] Compare with SimpleMoE (scikit-learn version)

- [ ] **Advanced Features**
  - [ ] Sequence length parameter tuning
  - [ ] Batch training
  - [ ] Learning rate scheduling
  - [ ] Early stopping with validation

### Week 5: Final Evaluation & Documentation (August 16-22, 2026)

- [ ] **Final Evaluation**
  - [ ] Compare all models (baselines + HMM + MoE variants)
  - [ ] Generate final results report
  - [ ] Create performance tables
  - [ ] Write research summary

- [ ] **Documentation**
  - [ ] Complete README with usage examples
  - [ ] API documentation with docstrings
  - [ ] Create Jupyter notebooks for exploration
  - [ ] Write blog post/medium article

- [ ] **Final Deliverables**
  - [ ] Results report (PDF/Markdown)
  - [ ] Visualization suite
  - [ ] Reproducible code package
  - [ ] Presentation slides

---

## 📋 OPTIONAL TASKS (Stretch Goals)

### Data Enhancements
- [ ] Add Ken French Data Library factors (HML, UMD, SMB, QMJ, BAB)
- [ ] Add FRED macroeconomic indicators
- [ ] Test on international factor data (MSCI World ex-US)
- [ ] Add sentiment indicators (VIX, put/call ratio)

### Model Enhancements
- [ ] Add XGBoost baseline
- [ ] Add Bayesian regression with regime switching
- [ ] Add online learning for streaming data
- [ ] Add ensemble of MoE models

### Infrastructure
- [ ] Add DVC for data version control
- [ ] Add MLflow for experiment tracking
- [ ] Create Docker container for reproducibility
- [ ] Set up CI/CD pipeline (GitHub Actions)

### Deployment
- [ ] Create simple web dashboard (Streamlit/Dash)
- [ ] Build API for real-time factor allocation
- [ ] Create automated report generation

---

## 🐛 BUG TRACKER

### Open Issues

| ID | Description | Priority | Status |
|----|-------------|----------|--------|
| #1 | Unicode emoji causing logging error on Windows | Low | Open |
| #2 | Investment metrics based on actual returns, not predictions | High | To Fix |

### Fixed Issues

| ID | Description | Fix Date |
|----|-------------|----------|
| #3 | 'Close' column KeyError with yfinance MultiIndex | July 25, 2026 |
| #4 | Models predicting single output instead of multi-output | July 25, 2026 |
| #5 | Missing tqdm dependency | July 25, 2026 |

---

## 💡 NOTES & OBSERVATIONS

### Research Notes
- MoE shows 4% improvement in RMSE over Rolling Average
- VIX is the hardest factor to predict (RMSE ~26-27%)
- USMV (Low Volatility) is the easiest factor to predict (RMSE ~4%)
- All models show similar Sharpe ratio because we're using actual returns for investment metrics

### Technical Notes
- Data range: 2013-08 to 2026-07 (156 months)
- Backtest window: 2019-08 to 2026-06 (83 predictions)
- Features: 28 (lagged returns + macro)
- Factors: 6 (SPY, IWD, MTUM, QUAL, USMV, VIX)

### Lessons Learned
1. Baseline first approach works well - establishes clear comparison
2. Simple MoE with linear experts is effective with limited data
3. yfinance with auto_adjust=True creates MultiIndex columns
4. Multi-output support is essential for factor prediction

---

## 🔗 QUICK LINKS

### Project Resources
- Repository: https://github.com/kira-ml/mixture-of-experts-factor-timing
- Problem Framing: `problem_framing.md`
- Results: `results/` directory

### Key Scripts
- Main orchestrator: `python main.py --run-all`
- Data only: `python main.py --download-data`
- Backtest only: `python main.py --backtest`

### Custom Runs
```bash
# Run with custom parameters
python main.py --run-all --start-date 2000-01-01 --min-train 84 --lags 1 3 6 12 24

# Run specific models
python main.py --run-all --models persistence rolling_avg moe

# Run with custom MoE parameters
python main.py --run-all --models moe
```

---

## 📊 METRICS TRACKER

| Date | Model | RMSE | MAE | Sharpe |
|------|-------|------|-----|--------|
| 2026-07-25 | Persistence | 17.8404 | 9.8470 | 1.7667 |
| 2026-07-25 | Rolling Avg | 12.2288 | 6.7109 | 1.7667 |
| 2026-07-25 | Momentum | 12.3836 | 6.7823 | 1.7667 |
| 2026-07-25 | Linear | 14.9084 | 8.2372 | 1.7667 |
| 2026-07-25 | RF | 13.7637 | 7.3650 | 1.7667 |
| 2026-07-25 | **MoE (K=3)** | **11.7350** | **6.6553** | 1.7667 |

---

## ✨ SUCCESS CRITERIA CHECKLIST

### Primary (Predictive Performance)
- [x] Out-of-sample log-likelihood (MoE implemented)
- [ ] Negative log-likelihood evaluation
- [x] RMSE comparison across models
- [x] MAE comparison across models

### Secondary (Investment Performance)
- [ ] Sharpe ratio from model-based allocation
- [ ] Maximum drawdown from model-based allocation
- [ ] Calmar ratio from model-based allocation
- [ ] Turnover measurement
- [ ] Transaction-cost-adjusted returns

### Interpretability
- [x] Regime probabilities from MoE
- [ ] Regime alignment with economic events
- [ ] Expert characteristic analysis

---

## 🎓 REFERENCES

1. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

2. Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.

3. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79-87.

4. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

---

## 📝 DAILY CHECK-IN TEMPLATE

### [Date] - [Day Number]

#### Completed
- 

#### In Progress
- 

#### Blocked By
- 

#### Next Steps
- 

#### Notes/Observations
- 

---

**Last Updated:** July 25, 2026  
**Project Status:** 🟢 Active Development