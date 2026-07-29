# Problem Framing: Regime-Aware Factor Timing

---

## Overview

This document frames the research problem for the `moe-factor-timing` project. It defines the core research question, problem motivation, machine learning formulation, success criteria, and evaluation strategy. The framing is designed to be practical, technically sound, and valuable as an open-source contribution to the quantitative finance community.

---

## 1. Research Question

**Primary Question:**

> Can a probabilistic model that explicitly represents uncertainty over latent economic regimes improve out-of-sample factor allocation performance compared to simpler deterministic approaches?

**Secondary Questions:**

1. How does a Mixture of Experts (MoE) model compare to linear and non-linear machine learning baselines for multi-factor return prediction?
2. Do macroeconomic indicators provide incremental predictive value beyond historical factor returns and market volatility in a regime-aware framework?
3. Are the learned regimes economically interpretable, and do they exhibit stability over time?

---

## 2. Problem Definition

### The Core Challenge

Equity factor premiums—such as Value, Momentum, Quality, and Low Volatility—are central to modern portfolio construction. However, these premiums are not stable over time; they exhibit significant time-variation that appears related to macroeconomic conditions. This creates a practical challenge: how should an investor allocate across factors when the future performance of each factor is uncertain and dependent on latent economic states?

### Why This Problem Matters

Factor investing has grown significantly, but factor performance cycles are difficult to time. Investors face a practical dilemma:

| Approach | Advantage | Disadvantage |
|----------|-----------|--------------|
| Static factor allocation | Diversification, low turnover | Prolonged drawdowns, missed opportunities |
| Aggressive factor timing | Potential for higher returns | High implementation risk; often fails |
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
| Historical factor returns | Monthly returns for months $t-L$ through $t$ | yfinance |
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

The model produces **distributional** predictions, not point estimates. This allows downstream allocation to account for prediction uncertainty when making investment decisions. Importantly, the model is **regime-aware** (it estimates the probability of being in different latent states), rather than enforcing a hard regime-switching rule.

---

## 4. Models

To ensure a rigorous comparison, we evaluate models across three levels of complexity.

### Level 1: Non-Machine Learning Heuristics (Baselines)

| Model | Description | Why Included |
|-------|-------------|--------------|
| **Persistence Forecast** | Next month's return equals current month's return | Simplest benchmark; establishes minimum performance threshold |
| **Rolling Average** | Next month's return equals average of last 12 months | Captures medium-term trends; common practitioner heuristic |

### Level 2: Standard Machine Learning Baselines

| Model | Description | Why Included |
|-------|-------------|--------------|
| **Linear Regression** | Linear model with lagged returns and macro features | Standard statistical baseline; establishes linear benchmark |
| **Random Forest** | Non-linear ensemble with 100 trees, max depth 10 | Captures non-linear relationships; robust to overfitting |

### Level 3: Advanced Method (Primary Contribution)

| Model | Description | Why Included |
|-------|-------------|--------------|
| **Mixture of Experts (MoE)** | Softmax gating network + linear experts per regime, trained via EM | Explicitly models regime uncertainty; primary research contribution |

---

## 5. Evaluation Strategy

### Validation Approach

We use **expanding window cross-validation** (time-series cross-validation):

| Parameter | Value |
|-----------|-------|
| Window Type | Expanding (growing training set) |
| Minimum Training Size | 132 months (optimal) |
| Test Size | 1 month |
| Out-of-Sample Predictions | 24 months (2024-07 to 2026-06) |

**Rationale:** Expanding window cross-validation prevents look-ahead bias and mimics real-world deployment. Given the use of 96 macroeconomic features, a minimum training size of 132 months is required to stabilize the MoE's Expectation-Maximization algorithm and avoid singular covariance matrices.

### Evaluation Metrics

We evaluate models on two distinct dimensions to capture the trade-off between predictive accuracy and investment utility.

**Dimension 1: Predictive Accuracy**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Lower is better; penalizes large errors |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Lower is better; interpretable in same units |

**Dimension 2: Investment Performance**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Sharpe Ratio | $\frac{\text{Mean Excess Return}}{\text{Std Dev Return}} \times \sqrt{12}$ | Higher is better; risk-adjusted return |
| Annualized Return | $(1 + \text{Total Return})^{1/\text{Years}} - 1$ | Higher is better; compound growth |
| Maximum Drawdown | $\min_{t} \frac{\text{Cumulative}_t - \text{Max Cumulative}}{\text{Max Cumulative}}$ | Lower (less negative) is better |
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
| **Macroeconomic Data** | FRED | CPI, INDPRO, UNRATE, T10Y2Y, GS10, GS2 |

**Data Period:** 2013-08 to 2026-07 (156 months)

**Why These Datasets:**
- yfinance provides accessible, real-world data for factor proxies
- FRED provides standard macroeconomic indicators used in academic research
- Monthly frequency aligns with factor returns
- Freely available and reproducible

---

## 7. Experimental Design

### Experiment 1: Model Comparison (Primary)
Compare all models (heuristics, ML baselines, and MoE) using expanding window backtest.

**Hypothesis:** MoE will outperform baselines on risk-adjusted investment metrics (Sharpe Ratio).

### Experiment 2: Feature Set Comparison
Compare VIX-only features (28 features) vs. FRED-enhanced features (96 features).

**Hypothesis:** FRED indicators will provide incremental predictive value beyond VIX.

### Experiment 3: Regime Sensitivity Analysis
Test different numbers of experts (K=2, 3, 4, 5) to identify the optimal number of latent regimes.

**Hypothesis:** Optimal K will be between 3-5 for this data.

### Experiment 4: Robustness Checks (Secondary)
Test the stability of the optimal MoE configuration under varying training window sizes (e.g., 120, 132, 144 months).

**Hypothesis:** The model will require a minimum of 132 months of training data to maintain stable predictions.

---

## 8. Success Criteria

### Primary Success Criteria

| Criterion | Definition |
|-----------|------------|
| **Model Outperforms Baselines** | MoE achieves a higher Sharpe ratio than Linear Regression and Rolling Average |
| **Positive Sharpe Ratio** | The MoE strategy should achieve a Sharpe ratio > 0 |
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
4. **Feature dimensionality trade-off:** The use of 96 FRED features requires a sufficiently long training window (132 months) to maintain model stability.

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
| 3.0 | 2026-07-30 | Refined scope: changed "regime-switching" to "regime-aware"; aligned out-of-sample window with data constraints; added robustness checks |

---

## 14. License

This project is released under the [MIT License](../LICENSE).

---

*For questions or contributions, please open an issue or pull request on GitHub.*