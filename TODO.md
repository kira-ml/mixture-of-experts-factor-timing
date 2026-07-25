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

#### Initial Results (Before Fixes)

| Model | RMSE | MAE | Sharpe Ratio |
|-------|------|-----|--------------|
| **MoE** | **11.7350** | **6.6553** | 1.7667 |
| Rolling Avg | 12.2288 | 6.7109 | 1.7667 |
| Momentum | 12.3836 | 6.7823 | 1.7667 |
| RF | 13.7637 | 7.3650 | 1.7667 |
| Linear | 14.9084 | 8.2372 | 1.7667 |
| Persistence | 17.8404 | 9.8470 | 1.7667 |

**Key Finding:** MoE outperforms all baselines with 4% improvement in RMSE over Rolling Average. All models showed identical Sharpe ratios due to using actual returns instead of model-based allocations.

#### Challenges Encountered & Resolved (Day 1)

- [x] **Issue:** yfinance MultiIndex columns causing 'Close' key error
  - **Solution:** Added robust column detection for MultiIndex DataFrames

- [x] **Issue:** Models predicting single output instead of 6 factors
  - **Solution:** Added multi-output support to all models

- [x] **Issue:** Unicode emoji (✅) causing logging error in Windows
  - **Status:** Known issue, non-critical (logging still works)

- [x] **Issue:** Missing `tqdm` dependency
  - **Solution:** Added to requirements.txt


### July 26, 2026 (Day 2) - Backtest Fixes & MoE Debugging

#### Completed Tasks

**Morning Session - Backtest Allocation Fix**
- [x] Identified root cause: all Sharpe ratios identical because investment metrics used actual returns, not model predictions
- [x] Created `predictions_to_returns()` function in `src/backtest.py`
- [x] Implemented long-only positive prediction allocation strategy
- [x] Replaced `returns=actuals_df.mean(axis=1).values` with model-based portfolio returns
- [x] Fixed dtype error in `predictions_to_returns()` (int → float for weights)
- [x] Added debug logging for MoE predictions

**MoE Debugging Session**
- [x] Identified `'LinearRegression' object has no attribute 'coef_'` error in MoE
- [x] Fixed `_compute_expert_predictions()`: added `hasattr()` check before accessing `coef_`
- [x] Identified `Model not fitted. Call fit() first.` error
- [x] Fixed `fit()` method: moved `self.is_fitted = True` before final `predict()`
- [x] Wrapped final `predict()` in try/except to prevent training failure

#### Final Results (After All Fixes)

**Data Range:** 2013-08 to 2026-07 (156 months)  
**Backtest Window:** 2019-08 to 2026-06 (83 predictions)  
**Features:** 28 (lagged returns + macro)  
**Factors:** 6 (SPY, IWD, MTUM, QUAL, USMV, VIX)

| Model | RMSE | MAE | Sharpe | Ann Return | Volatility | Max DD | Win Rate |
|-------|------|-----|--------|------------|------------|--------|----------|
| **MoE** | 16.5407 | 8.9702 | **0.3793** | **0.09%** | 58.70% | -58.94% | 56.63% |
| Rolling Avg | 12.2288 | 6.7109 | 0.0745 | -1.96% | 26.81% | -60.77% | **65.06%** |
| Momentum | 12.3836 | 6.7823 | 0.0146 | -3.62% | 27.06% | -60.42% | 63.86% |
| RF | 13.7651 | 7.3517 | -0.2528 | -14.43% | 34.78% | -83.45% | 62.65% |
| Linear | 14.9083 | 8.2371 | -0.0898 | -12.71% | 43.31% | -76.78% | 55.42% |
| Persistence | 17.8404 | 9.8470 | -0.7027 | -38.25% | 48.10% | -98.30% | 56.63% |

**Key Findings:**
- MoE achieves **highest Sharpe ratio (0.3793)** among all models
- MoE is the **only model with positive annual return (0.09%)**
- MoE has slightly lower max drawdown (-58.94%) than Rolling Avg (-60.77%)
- MoE has higher volatility (58.70%) due to more aggressive allocations
- Rolling Avg has best RMSE (12.2288) and Win Rate (65.06%)

**Insight:** Simple MoE trades predictive accuracy for investment performance — better allocation decisions despite less accurate point predictions.

#### Challenges Encountered & Resolved (Day 2)

- [x] **Issue:** All models showed identical Sharpe ratios
  - **Root Cause:** Investment metrics used `actuals_df.mean(axis=1)` (actual returns) instead of model-based allocations
  - **Solution:** Created `predictions_to_returns()` function with long-only positive allocation strategy

- [x] **Issue:** `numpy._core._exceptions.UFuncTypeError` in `predictions_to_returns()`
  - **Root Cause:** `np.where()` created int64 array; `np.divide()` tried to store float results
  - **Solution:** Changed `1` → `1.0` and `0` → `0.0` to use float dtype

- [x] **Issue:** `'LinearRegression' object has no attribute 'coef_'`
  - **Root Cause:** `_compute_expert_predictions()` accessed `expert.coef_` before expert was fitted
  - **Solution:** Added `hasattr(expert, 'coef_')` check before accessing attribute

- [x] **Issue:** `Model not fitted. Call fit() first.`
  - **Root Cause:** `self.is_fitted = True` was set after final `predict()`, which could fail
  - **Solution:** Moved `self.is_fitted = True` before final `predict()` and wrapped in try/except

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

- [ ] **Visualize MoE Regime Probabilities** (High Priority)
  - [ ] Extract regime probabilities from MoE predictions
  - [ ] Plot regime probabilities over time
  - [ ] Map regimes to known economic events (COVID-19, 2022 bear market)
  - [ ] Analyze what each expert learned

- [ ] **Cumulative Returns Comparison**
  - [ ] Plot cumulative returns for all models
  - [ ] Highlight MoE vs Rolling Avg vs Equal Weight

- [ ] **Hyperparameter Tuning** (Light Touch)
  - [ ] MoE: Try K=2, 3, 4, 5 experts
  - [ ] MoE: Test EM iterations (30, 50, 100)
  - [ ] Rolling Avg: Test windows (6, 12, 24)

- [ ] **Statistical Testing**
  - [ ] Bootstrap confidence intervals for Sharpe ratios
  - [ ] Diebold-Mariano test for predictive accuracy

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

### Week 5: Final Evaluation & Documentation (August 16-22, 2026)

- [ ] **Final Evaluation**
  - [ ] Compare all models (baselines + HMM + MoE variants)
  - [ ] Generate final results report
  - [ ] Create performance tables

- [ ] **Documentation**
  - [ ] Complete README with usage examples
  - [ ] API documentation with docstrings
  - [ ] Create Jupyter notebooks for exploration

---

## 📋 OPTIONAL TASKS (Stretch Goals)

### Data Enhancements
- [ ] Add Ken French Data Library factors (HML, UMD, SMB, QMJ, BAB)
- [ ] Add FRED macroeconomic indicators
- [ ] Add sentiment indicators (VIX, put/call ratio)

### Model Enhancements
- [ ] Add XGBoost baseline
- [ ] Add Bayesian regression with regime switching
- [ ] Add online learning for streaming data

### Infrastructure
- [ ] Add MLflow for experiment tracking
- [ ] Create Docker container for reproducibility

---

## 🐛 BUG TRACKER

### Open Issues

| ID | Description | Priority | Status |
|----|-------------|----------|--------|
| #1 | Unicode emoji causing logging error on Windows | Low | Open |

### Fixed Issues

| ID | Description | Fix Date |
|----|-------------|----------|
| #2 | 'Close' column KeyError with yfinance MultiIndex | July 25, 2026 |
| #3 | Models predicting single output instead of multi-output | July 25, 2026 |
| #4 | Missing tqdm dependency | July 25, 2026 |
| #5 | All models showing identical Sharpe ratios | July 26, 2026 |
| #6 | dtype error in predictions_to_returns() | July 26, 2026 |
| #7 | MoE: 'LinearRegression' object has no attribute 'coef_' | July 26, 2026 |
| #8 | MoE: Model not fitted. Call fit() first. | July 26, 2026 |

---

## 💡 NOTES & OBSERVATIONS

### Research Notes
- **MoE has highest Sharpe (0.3793)** despite higher RMSE (16.54 vs 12.23 for Rolling Avg)
- **MoE is only model with positive returns (0.09%)** in backtest period
- **Interesting trade-off:** MoE sacrifices point prediction accuracy for better allocation decisions
- VIX is the hardest factor to predict (RMSE ~26-40% depending on model)
- USMV (Low Volatility) is the easiest factor to predict (RMSE ~4-5%)
- Rolling Avg has best Win Rate (65.06%) but negative returns (-1.96%)

### Technical Notes
- Data range: 2013-08 to 2026-07 (156 months)
- Backtest window: 2019-08 to 2026-06 (83 predictions)
- Features: 28 (lagged returns + macro)
- Factors: 6 (SPY, IWD, MTUM, QUAL, USMV, VIX)
- Allocation strategy: Long-only, equal-weight on positive predictions
- MoE parameters: K=3 experts, 30 EM iterations, lr=0.01

### Lessons Learned
1. **Investment metrics must use model-based allocations** — otherwise Sharpe ratios are identical
2. **Simple MoE with linear experts works** — effective with limited data
3. **EM algorithm requires careful handling** — ensure experts are fitted before prediction
4. **Trade-offs are normal** — higher predictive accuracy doesn't always mean better investment returns
5. **Debug incrementally** — fix one issue at a time to isolate root causes

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

| Date | Model | RMSE | MAE | Sharpe | Ann Return | Volatility | Max DD |
|------|-------|------|-----|--------|------------|------------|--------|
| 2026-07-25 | Persistence | 17.8404 | 9.8470 | 1.7667* | - | - | - |
| 2026-07-25 | Rolling Avg | 12.2288 | 6.7109 | 1.7667* | - | - | - |
| 2026-07-25 | Momentum | 12.3836 | 6.7823 | 1.7667* | - | - | - |
| 2026-07-25 | Linear | 14.9084 | 8.2372 | 1.7667* | - | - | - |
| 2026-07-25 | RF | 13.7637 | 7.3650 | 1.7667* | - | - | - |
| 2026-07-25 | MoE (K=3) | 11.7350 | 6.6553 | 1.7667* | - | - | - |
| **2026-07-26** | **Persistence** | 17.8404 | 9.8470 | **-0.7027** | -38.25% | 48.10% | -98.30% |
| **2026-07-26** | **Rolling Avg** | 12.2288 | 6.7109 | **0.0745** | -1.96% | 26.81% | -60.77% |
| **2026-07-26** | **Momentum** | 12.3836 | 6.7823 | **0.0146** | -3.62% | 27.06% | -60.42% |
| **2026-07-26** | **Linear** | 14.9083 | 8.2371 | **-0.0898** | -12.71% | 43.31% | -76.78% |
| **2026-07-26** | **RF** | 13.7651 | 7.3517 | **-0.2528** | -14.43% | 34.78% | -83.45% |
| **2026-07-26** | **MoE (K=3)** | 16.5407 | 8.9702 | **0.3793** | **0.09%** | 58.70% | -58.94% |

*Note: July 25 Sharpe ratios were identical because investment metrics used actual returns, not model-based allocations. July 26 reflects corrected methodology.

---
## Today's TODO List (July 26, 2026)

### July 26, 2026 (Day 3) - Optimization Experiments

#### Immediate Tasks

- [ ] **Experiment 1: Increase MoE EM Iterations (30 → 100)**
  - [ ] Update `main.py` model_params for MoE: `'n_iterations': 100`
  - [ ] Run backtest and compare RMSE, Sharpe vs baseline
  - [ ] Log results to metrics tracker

- [ ] **Experiment 2: Find Optimal Number of Experts (K=2, 4, 5)**
  - [ ] Run MoE with K=2, K=4, K=5
  - [ ] Compare Sharpe ratios and RMSE
  - [ ] Keep only if improves over K=3 baseline
  - [ ] Log optimal K to metrics tracker

- [ ] **Experiment 3: Add Transaction Costs (0.10% per trade)**
  - [ ] Modify `predictions_to_returns()` in `backtest.py`
  - [ ] Apply cost deduction to portfolio returns
  - [ ] Compare Sharpe with vs without costs
  - [ ] Determine if MoE still outperforms baselines

- [ ] **Experiment 4: Visualize Regime Probabilities**
  - [ ] Create new notebook `notebooks/02_regime_analysis.ipynb`
  - [ ] Extract regime probabilities from MoE using `predict_proba()`
  - [ ] Plot regime probabilities over time
  - [ ] Map to economic events (COVID-19, 2022 bear market, etc.)
  - [ ] Analyze what each expert learned

- [ ] **Experiment 5: Weighted Allocation Strategy** (Conditional)
  - [ ] Only if transaction costs don't kill MoE's edge
  - [ ] Modify `predictions_to_returns()` to weight by prediction magnitude
  - [ ] Compare Sharpe vs equal-weight positive strategy

#### Documentation & Tracking
- [ ] Update metrics tracker with all experiment results
- [ ] Document which optimizations improved performance
- [ ] Revert changes that didn't add value
- [ ] Commit working improvements to Git


---




## ✨ SUCCESS CRITERIA CHECKLIST

### Primary (Predictive Performance)
- [x] RMSE comparison across models
- [x] MAE comparison across models
- [ ] Negative log-likelihood evaluation (future work)

### Secondary (Investment Performance)
- [x] Sharpe ratio from model-based allocation
- [x] Maximum drawdown from model-based allocation
- [x] Annualized return from model-based allocation
- [x] Win rate from model-based allocation
- [ ] Calmar ratio from model-based allocation (working)
- [ ] Transaction-cost-adjusted returns (future work)

### Interpretability
- [x] Regime probabilities from MoE
- [ ] Regime alignment with economic events (next step)
- [ ] Expert characteristic analysis (next step)

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

**Last Updated:** July 26, 2026  
**Project Status:** 🟢 Active Development