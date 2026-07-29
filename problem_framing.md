# Problem Framing: Regime-Switching Factor Timing

---

## Overview

This document frames the research problem for the `moe-factor-timing` project. It defines the core research question, problem motivation, machine learning formulation, success criteria, and evaluation strategy. The framing is designed to be practical, technically sound, and valuable as an open-source contribution.

---

## 1. Research Question

**Primary Question:**

> Can a probabilistic model that explicitly represents uncertainty over latent economic regimes improve out-of-sample factor timing performance compared to simpler deterministic approaches?

**Secondary Questions:**

1. How does a Mixture of Experts model compare to linear and non-linear baselines for factor return prediction?
2. Do macroeconomic indicators provide incremental predictive value beyond historical factor returns and market volatility?
3. Are the learned regimes economically interpretable and stable over time?

---

## 2. Problem Definition

### The Core Challenge

Equity factor premiums—such as Value, Momentum, Quality, and Low Volatility—are central to modern portfolio construction (Fama & French, 1993; Asness et al., 2013). However, these premiums are not stable over time. Value can underperform for extended periods, and momentum can experience sudden reversals. This creates a practical challenge for investors: how to allocate across factors when the future performance of each factor is uncertain and regime-dependent.

### Why This Problem Matters

Factor investing has grown into a multi-trillion-dollar industry, but factor premiums are not stable over time. Investors face a practical dilemma:

| Approach | Advantage | Disadvantage |
|----------|-----------|--------------|
| Static factor allocation | Diversification, low turnover | Prolonged drawdowns, missed opportunities |
| Aggressive factor timing | Potential for higher returns | Implementation risk, can fail spectacularly |
| Regime-aware allocation | Potential to adapt to changing conditions | Requires reliable regime detection |

This project addresses the intermediate question of whether probabilistic regime awareness can improve allocation decisions without overfitting to historical patterns.

### What This Project Does

We develop and compare methods that explicitly model latent economic regimes probabilistically, using factor return data and macroeconomic indicators. We test whether probabilistic regime representation improves the risk-adjusted performance of dynamic factor allocation relative to simpler baselines.

---

## 3. Machine Learning Formulation

### Task Type

**Probabilistic time-series forecasting** with an implicit decision-making component (portfolio allocation).

### Inputs ($X_t$)

| Feature Type | Description | Source |
|--------------|-------------|--------|
| Historical factor returns | Monthly returns for months $t-L$ through $t$ | yfinance / FRED |
| Macroeconomic indicators | VIX, CPI, Industrial Production, Unemployment, Term Spread | FRED |
| Derived features | Lagged returns, rolling volatility, transformations | Engineered |

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

### Evaluation Framework

We evaluate models on two dimensions:

**Dimension 1: Predictive Accuracy** — How well does the model predict future factor returns?
- Primary metric: RMSE (Root Mean Squared Error)
- Secondary metric: MAE (Mean Absolute Error)

**Dimension 2: Investment Performance** — How well do predictions translate into portfolio performance?
- Primary metric: Sharpe Ratio (risk-adjusted return)
- Secondary metrics: Annualized Return, Maximum Drawdown, Calmar Ratio, Win Rate

This dual evaluation is critical because a model can have high predictive accuracy but poor investment performance (if predictions are noisy in the wrong direction), or vice versa.

---

## 4. Models

### Baseline Models

| Model | Description | Why Included |
|-------|-------------|--------------|
| **Persistence Forecast** | Next month's return equals current month's return | Simplest possible benchmark; establishes minimum performance |
| **Rolling Average** | Next month's return equals the average of the last 12 months | Captures medium-term trends; common practitioner heuristic |
| **Linear Regression** | Linear model with lagged returns and macro features | Standard statistical baseline; establishes linear benchmark |

### Machine Learning Baselines

| Model | Description | Why Included |
|-------|-------------|--------------|
| **Random Forest** | Non-linear ensemble with 100 trees, max depth 10 | Captures non-linear relationships; robust to overfitting |
| **Momentum-Based Timing** | Allocate based on 12-month factor momentum | Common practitioner approach; establishes heuristic baseline |

### Advanced Model

| Model | Description | Why Included |
|-------|-------------|--------------|
| **Mixture of Experts (MoE)** | Softmax gating network + expert models per regime | Explicitly models regime uncertainty; primary research contribution |

---

## 5. Evaluation Strategy

### Validation Approach

We use **expanding window cross-validation** (also known as time-series cross-validation):

| Parameter | Value |
|-----------|-------|
| Window Type | Expanding (growing training set) |
| Minimum Training Size | 60-132 months (tested) |
| Test Size | 1 month |
| Predictions | 83+ months (2019-2026) |

**Rationale:** Expanding window prevents look-ahead bias and mimics real-world deployment where historical data accumulates over time.

### Evaluation Metrics

**Predictive Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Lower is better; penalizes large errors |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Lower is better; interpretable in same units |

**Investment Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Sharpe Ratio | $\frac{\text{Mean Excess Return}}{\text{Std Dev Return}} \times \sqrt{12}$ | Higher is better; risk-adjusted return |
| Annualized Return | $(1 + \text{Total Return})^{1/\text{Years}} - 1$ | Higher is better; compound growth |
| Maximum Drawdown | $\min_{t} \frac{\text{Cumulative}_t - \text{Max Cumulative}}{\text{Max Cumulative}}$ | Lower is better (less negative) |
| Calmar Ratio | $\frac{\text{Annualized Return}}{\|\text{Maximum Drawdown}\|}$ | Higher is better; return per unit of worst-case risk |
| Win Rate | $\frac{\text{Number of Positive Months}}{\text{Total Months}}$ | Higher is better; consistency |

### Transaction Cost Assumptions

We assume **10 basis points (0.10%)** per trade, which is realistic for institutional investors trading liquid ETFs.

### Allocation Strategy

We use a **magnitude-weighted long-only strategy**:
1. For each month, allocate to factors with positive predicted returns
2. Weight positions proportionally to the magnitude of the prediction
3. No short positions

**Rationale:** This strategy is practical for institutional investors and avoids the complexity of short-selling.

---

## 6. Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| **Equity Factors (Primary)** | yfinance (ETF proxies) | SPY, IWD, MTUM, QUAL, USMV, VIX |
| **Macroeconomic Data** | FRED (Federal Reserve Economic Data) | CPI, INDPRO, UNRATE, T10Y2Y, GS10, GS2 |

**Why These Datasets:**
- yfinance provides accessible, real-world data for factor proxies
- FRED provides standard macroeconomic indicators used in academic research
- Monthly frequency aligns with factor returns
- Freely available and reproducible

**Data Period:** 2013-08 to 2026-07 (156 months)

---

## 7. Experimental Design

### Experiment 1: Model Comparison (Primary)

Compare all models (baselines + MoE) using expanding window backtest.

**Hypothesis:** MoE will outperform baselines on investment metrics.

### Experiment 2: Macro Indicator Comparison

Compare VIX-only vs FRED-enhanced features.

**Hypothesis:** FRED indicators will provide incremental predictive value beyond VIX.

### Experiment 3: Regime Sensitivity Analysis

Test different numbers of experts (K=2,3,4,5,6,8).

**Hypothesis:** Optimal K will be between 3-5 for this data.

---

## 8. Success Criteria

### Primary Success Criteria

| Criterion | Definition |
|-----------|------------|
| **Model Outperforms Baselines** | MoE achieves higher Sharpe ratio than Linear Regression and Rolling Average |
| **Positive Sharpe Ratio** | All models should achieve > 0 Sharpe (risk-adjusted positive returns) |
| **Regime Interpretability** | Learned regimes should have distinct return/volatility characteristics |

### Secondary Success Criteria

| Criterion | Definition |
|-----------|------------|
| **Transaction Cost Robustness** | Performance remains positive after 10 bps transaction costs |
| **Win Rate > 50%** | Models should have > 50% monthly win rate |
| **Interpretable Regimes** | Regimes should align with known economic conditions |

---

## 9. Assumptions and Constraints

### Methodological Assumptions

1. **Future regimes resemble past regimes** in structure (if not timing)
2. **Macroeconomic indicators** capture relevant regime information
3. **Monthly rebalancing** is practical for institutional investors
4. **Transaction costs** of 10 bps are realistic for liquid ETFs

### Data Constraints

1. **Limited sample size** (156 months) limits model complexity
2. **US-only data** may not generalize to other markets
3. **ETF proxies** may not perfectly capture factor exposure
4. **Macro data revisions** (vintage effects) are not modeled

### Practical Constraints

1. **Long-only allocation** only (no short-selling)
2. **Monthly frequency** only (not daily or intraday)
3. **No leverage** (1x exposure only)

---

## 10. Deliverables

| Deliverable | Description | Format |
|-------------|-------------|--------|
| **Data Pipeline** | End-to-end data loading and preprocessing | Python module |
| **Model Implementations** | All models with consistent API | Python classes |
| **Backtesting Framework** | Expanding window cross-validation | Python module |
| **Evaluation Module** | Predictive + investment metrics | Python module |
| **Visualization Suite** | Regime plots, performance charts | Python module + PNG |
| **Results Report** | Summary of findings | CSV + Markdown |
| **README** | Project overview and usage | Markdown |

---

## 11. Open-Source Contribution

### Value to the Community

1. **Benchmark** comparing multiple approaches on real data
2. **Implementation** of probabilistic forecasting with mixture density outputs
3. **Transparent** backtesting with realistic assumptions
4. **Interpretable** regime identification framework
5. **Educational** value for practitioners learning factor timing

### Community Standards

| Standard | Status |
|----------|--------|
| Reproducible | ✅ Full code, data sources documented |
| Transparent | ✅ No proprietary data or black-box models |
| Extensible | ✅ Modular code structure |
| Educational | ✅ Clear implementations and documentation |

---

## 12. References

1. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

2. Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.

3. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

4. Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68.

5. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79-87.

6. Shih, W. (2020). *Machine Learning for Factor Investing*. CFA Institute Research Foundation.

---

## 13. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-07-25 | Initial problem framing |
| 2.0 | 2026-07-29 | Optimized framing: added evaluation framework, success criteria, experimental design |

---

## 14. License

This project is released under the [MIT License](../LICENSE).

---

*For questions or contributions, please open an issue or pull request on GitHub.*