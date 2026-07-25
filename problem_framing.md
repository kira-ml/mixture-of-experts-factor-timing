# Problem Framing: Regime-Switching Factor Timing

---

## Overview

This document frames the research problem for the `moe-factor-timing` project. It defines the core research question, problem motivation, machine learning formulation, success criteria, and evaluation strategy.

---

## 1. Research Question

*Can a probabilistic model that explicitly represents uncertainty over economic regimes improve out-of-sample factor timing performance compared to simpler deterministic approaches that either ignore regimes entirely or assign binary regime labels?*

---

## 2. Problem Definition

### The Core Challenge

Equity factor premiums (Value, Momentum, Quality, Low Volatility) exhibit pronounced time-variation that appears related to macroeconomic conditions. However, the relationship is noisy, regimes are not directly observable, and investors face significant uncertainty about which regime currently prevails.

Most practical factor timing approaches either:

1. **Ignore regimes entirely** — static allocation to factors
2. **Use heuristic rules** — binary regime classification based on thresholded macro indicators
3. **Apply short-term momentum** — timing factors based on their own recent performance

These approaches fail to represent regime uncertainty probabilistically, potentially leading to overconfident allocations during ambiguous periods.

### What This Project Does

We develop and compare methods that explicitly model latent economic regimes probabilistically, using factor return data and macroeconomic indicators. We test whether probabilistic regime representation improves the risk-adjusted performance of dynamic factor allocation relative to simpler baselines.

### Why This Matters

Factor investing has become a multi-trillion-dollar industry, but factor premiums are not stable over time. Investors face a practical dilemma:

- **Static factor allocation** provides diversification but suffers prolonged drawdowns
- **Aggressive factor timing** introduces implementation risk and can fail spectacularly

This project addresses the intermediate question of whether probabilistic regime awareness can improve allocation decisions without overfitting to historical patterns.

---

## 3. Machine Learning Formulation

### Inputs ($X_t$)

- **Macroeconomic indicators** at month $t$:
  - CPI (inflation)
  - Industrial production growth
  - Unemployment rate
  - Term spread (10Y-2Y)

- **Historical factor returns** for months $t-12$ through $t$:
  - HML (Value)
  - UMD (Momentum)
  - SMB (Size)
  - QMJ (Quality)
  - BAB (Low Volatility)

### Outputs ($Y_{t+1}$)

Vector of next-month returns for each of the $K$ equity factors:

$$\mathbf{y}_{t+1} = [r_{t+1}^{(1)}, r_{t+1}^{(2)}, ..., r_{t+1}^{(K)}]$$

### Prediction Task

Estimate the conditional distribution:

$$P(\mathbf{y}_{t+1} \mid \mathbf{x}_t, \mathbf{y}_{t-L:t})$$

where:
- $\mathbf{x}_t$ = macroeconomic features at time $t$
- $\mathbf{y}_{t-L:t}$ = historical factor returns

### Key Distinction

The model produces **distributional** predictions, not point estimates. This allows downstream allocation to account for prediction uncertainty when making investment decisions.

### Model Architecture

We implement a **Mixture of Experts (MoE)** model:

1. **Gating Network** — An LSTM that processes the input sequence and outputs probabilities over $K$ latent regimes:

   $$\pi_t = \text{softmax}(\text{LSTM}(\mathbf{x}_t, \mathbf{y}_{t-L:t}))$$

2. **Expert Networks** — Separate linear models for each regime, each predicting factor returns:

   $$\hat{\mathbf{y}}_{t+1}^{(k)} = \mathbf{W}^{(k)} \mathbf{z}_t + \mathbf{b}^{(k)}$$

3. **Mixture Output** — The final prediction is a probability-weighted combination:

   $$P(\mathbf{y}_{t+1}) = \sum_{k=1}^{K} \pi_t^{(k)} \cdot \mathcal{N}(\hat{\mathbf{y}}_{t+1}^{(k)}, \Sigma^{(k)})$$

---

## 4. Success Criteria

### Primary (Predictive Performance)

| Metric | Description |
|--------|-------------|
| **Out-of-sample log-likelihood** | Higher values indicate better probabilistic predictions |
| **Negative log-likelihood** | Primary loss function; evaluate on held-out test periods |

### Secondary (Investment Performance)

| Metric | Description |
|--------|-------------|
| **Sharpe ratio** | Annualized return divided by annualized volatility |
| **Maximum drawdown** | Largest peak-to-trough decline |
| **Calmar ratio** | Annualized return / maximum drawdown |
| **Turnover** | Average monthly portfolio churn |
| **Transaction-cost-adjusted returns** | Returns net of realistic trading costs |

### Interpretability

- Stability of learned regime assignments over time
- Alignment of regimes with known economic episodes (e.g., 2008 financial crisis, COVID-19 pandemic)
- Economic interpretability of regime characteristics

---

## 5. Evaluation Strategy

### Validation Approach

We use **time-series cross-validation with expanding window**:

| Period | Purpose |
|--------|---------|
| 1990-2000 | Training |
| 2001-2005 | Validation (hyperparameter tuning) |
| 2006-2010 | Test (includes 2008 financial crisis) |
| 2011-2015 | Test (post-crisis recovery) |
| 2016-2020 | Test (includes COVID-19 pandemic) |

**Rationale:** Expanding window prevents look-ahead bias and mimics real-world deployment where historical data accumulates over time.

### Model Comparison

We compare the following approaches:

**Baseline Models:**
1. **Linear Regression** — Predicts factor returns from macro variables and lagged factor returns
2. **Random Forest** — Non-linear ensemble model with same inputs
3. **Hidden Markov Model (HMM)** — Two-stage approach: identify regimes, then predict within each regime

**Additional Baselines:**
4. **Persistence Forecast** — Predict next month's return equals current month's return
5. **Momentum-Based Timing** — Allocate based on 12-month factor momentum
6. **Equal-Weighted Portfolio** — Static 1/K allocation to all factors

**Advanced Model:**
7. **Mixture of Experts (MoE)** — End-to-end trained with LSTM gating network

### Statistical Testing

- Compare log-likelihoods using **Diebold-Mariano test** for predictive accuracy
- Compare Sharpe ratios using **bootstrap** or **moving block bootstrap**
- Report **effect sizes**, not just statistical significance

### Robustness Checks

- Vary number of regimes ($K = 2, 3, 4, 5$)
- Use alternative macro feature sets
- Test on sub-periods to identify when each model performs best

---

## 6. Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| **Equity Factors** | Kenneth R. French Data Library | Monthly returns for HML, UMD, SMB, RMW, CMA, Mkt-RF |
| **Macroeconomic Data** | FRED (Federal Reserve Economic Data) | CPI (inflation), INDPRO (industrial production), UNRATE (unemployment), T10Y2Y (term spread), GS10 (10-year rate) |

**Why These Datasets:**
- Gold standard for factor research (French data)
- Clean, well-maintained, and freely available
- Monthly frequency aligns with factor returns
- Widely used in academic and practitioner research

### Data Preprocessing

1. Align monthly macro data with French factor return dates
2. Handle missing values via forward-fill (macro data) or deletion (factor data)
3. Standardize macro features to zero mean and unit variance
4. Create lagged factor return features ($t-12$ to $t$)

---

## 7. Expected Deliverables

1. **Data Pipeline** — Python code that aligns and synchronizes macro data with French factor returns

2. **Model Implementations** — PyTorch implementation of:
   - Linear regression baseline
   - Random forest baseline
   - HMM with regime-specific regressions
   - Recurrent Mixture of Experts model

3. **Backtesting Framework** — Realistic backtest with:
   - Expanding window cross-validation
   - Transaction cost assumptions
   - Performance attribution

4. **Visualization Suite** — Plots including:
   - Regime probabilities over time mapped to economic events
   - Cumulative returns of all strategies
   - Factor allocation time series
   - Performance metrics comparison

5. **Results Report** — Summary of findings including:
   - Which model performs best and under what conditions
   - Economic interpretation of learned regimes
   - Recommendations for practitioners

---

## 8. Research Scope and Boundaries

### In Scope

- Monthly frequency analysis (not daily or intraday)
- US equity factors and US macroeconomic data
- Long-only factor allocation (not long/short strategies)
- Five standard equity factors (HML, UMD, SMB, QMJ, BAB)
- Transparent, interpretable regime identification

### Out of Scope

- High-frequency or intraday timing
- International factor data
- Short-selling or leveraged strategies
- Deep learning architectures beyond LSTM-MoE
- Production-ready trading system

---

## 9. Limitations and Assumptions

### Technical Limitations

- Small feature set limits model complexity
- Monthly data provides limited training samples (~300 months)
- Regime changes may be gradual rather than discrete

### Methodological Assumptions

- Future regimes resemble past regimes in structure (if not timing)
- Macroeconomic indicators capture relevant regime information
- Monthly rebalancing is practical for institutional investors

### Data Limitations

- Factor data is academic (long/short portfolios), not investable products
- Macro data revisions (vintage effects) are not modeled
- Transaction costs are estimated, not from actual execution data

---

## 10. Open-Source Contribution

This project is designed to be:

- **Reproducible** — All code, data sources, and evaluation methods are documented
- **Transparent** — No proprietary data or black-box models
- **Extensible** — Modular code structure allows new models or features
- **Educational** — Clear implementations serve as learning resources

### Value to the Community

- Benchmark comparing multiple approaches on canonical data
- Implementation of probabilistic forecasting with mixture density outputs
- Transparent backtesting with realistic assumptions
- Interpretable regime identification framework

---

## 11. References

1. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

2. Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.

3. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

4. Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68.

5. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79-87.

6. Shih, W. (2020). *Machine Learning for Factor Investing*. CFA Institute Research Foundation.

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-07-25 | Initial problem framing |

---

## 13. License

This project is released under the [MIT License](../LICENSE).

---

*For questions or contributions, please open an issue or pull request on GitHub.*