# Problem Framing: Probabilistic Regime-Aware Factor Timing

---

## Overview

This document frames the research problem for the `moe-factor-timing` project. The core objective is to investigate whether probabilistic models that explicitly represent uncertainty over latent economic regimes can provide a robust framework for dynamic equity factor allocation. This project is designed as a practical, open-source contribution to the quantitative finance community, prioritizing methodological transparency and empirical reproducibility over aggressive claims of financial outperformance.

---

## 1. Research Question

**Primary Question:**

> Does a probabilistic model that explicitly represents uncertainty over latent economic regimes offer a practical and coherent framework for out-of-sample factor allocation, and how does its performance compare to simpler deterministic approaches?

**Secondary Questions:**

1. How does a Mixture of Experts (MoE) model compare to linear and non-linear machine learning baselines in a realistic, expanding-window backtesting setup?
2. Do macroeconomic indicators (e.g., from FRED) provide incremental, non-redundant predictive value beyond historical factor returns and market volatility in a regime-aware framework?
3. Are the learned latent regimes economically interpretable, and do they exhibit distinct and stable characteristics over time?

---

## 2. Problem Definition

### The Core Challenge

Equity factor premiums—such as Value, Momentum, Quality, and Low Volatility—are central to modern portfolio construction. However, these premiums are not stable over time; they exhibit significant time-variation that is widely believed to be related to macroeconomic conditions. This creates a practical challenge: how should an investor allocate across factors when the future performance of each factor is uncertain and dependent on latent, unobservable economic states?

### Why This Problem Matters for Open-Source Research

While quantitative finance has widely adopted factor investing, the practical problem of timing factor exposures remains unresolved. Investors face a trade-off between static allocation (which provides diversification but suffers prolonged drawdowns) and aggressive tactical timing (which offers higher potential returns but carries significant implementation risk and often fails in practice).

This project does not claim to solve factor timing. Instead, it provides a transparent, reproducible benchmark for evaluating a specific class of solution: **probabilistic regime-aware models**. By making the code, data, and methodology fully open-source, we contribute a practical baseline that the community can use, critique, and extend.

### What This Project Does

We develop and compare methods that explicitly model latent economic regimes probabilistically, using factor return data and macroeconomic indicators. We test whether probabilistic regime representation offers a coherent and computationally practical framework for dynamic factor allocation relative to simpler, non-probabilistic baselines.

---

## 3. Machine Learning Formulation

### Task Type

**Probabilistic time-series forecasting** with an implicit decision-making component (portfolio allocation). The model generates distributional predictions, which are then used to inform an allocation strategy.

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

The model produces **distributional** predictions, not point estimates. This allows the downstream allocation strategy to account for prediction uncertainty. Importantly, the model is **regime-aware** (it estimates the probability of being in different latent states), rather than enforcing a hard regime-switching rule. This probabilistic representation is the project's core methodological contribution.

---

## 4. Models

To ensure a rigorous and transparent comparison, we evaluate models across three levels of complexity.

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
| Out-of-Sample Period | 2019-08 to 2026-06 (83 monthly observations) |

**Rationale:** Expanding window cross-validation prevents look-ahead bias and mimics real-world deployment. Given the use of 96 macroeconomic features, a minimum training size of 132 months is required to stabilize the MoE's Expectation-Maximization algorithm and avoid singular covariance matrices.

### Evaluation Metrics

We evaluate models on two distinct dimensions to capture the inherent trade-off between predictive accuracy and investment utility.

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
1. For each month, allocate to factors with positive predicted returns.
2. Weight positions proportionally to the magnitude of the prediction.
3. No short positions.

**Rationale:** This strategy is practical for institutional investors and avoids the complexity of short-selling.

---

## 6. Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| **Equity Factors (Primary)** | yfinance (ETF proxies) | SPY, IWD, MTUM, QUAL, USMV, VIX |
| **Macroeconomic Data** | FRED | CPI, INDPRO, UNRATE, T10Y2Y, GS10, GS2 |

**Data Period:** 2013-08 to 2026-07 (156 months)

**Why These Datasets:**
- yfinance provides accessible, real-world data for factor proxies.
- FRED provides standard macroeconomic indicators used in academic research.
- Monthly frequency aligns with factor returns.
- All data is freely available and the pipeline is fully reproducible.

---

## 7. Experimental Design

### Experiment 1: Model Comparison (Primary)
Compare all models (heuristics, ML baselines, and MoE) using an expanding window backtest.

**Hypothesis:** The MoE model will provide a useful and distinct benchmark for regime-aware allocation, with performance characteristics that differ from deterministic baselines.

### Experiment 2: Feature Set Comparison
Compare VIX-only features (28 features) vs. FRED-enhanced features (96 features).

**Hypothesis:** FRED indicators will provide non-redundant predictive information, but the high dimensionality requires careful regularization and a sufficiently long training window.

### Experiment 3: Regime Sensitivity Analysis
Test different numbers of experts (K=2, 3, 4, 5) to identify the optimal number of latent regimes.

**Hypothesis:** The optimal K will be between 3-5 for this data, balancing model expressiveness with overfitting risk.

### Experiment 4: Robustness Checks (Secondary)
Test the stability of the optimal MoE configuration under varying training window sizes (e.g., 120, 132, 144 months).

**Hypothesis:** The model will require a minimum of 132 months of training data to maintain stable parameter estimates and reliable regime assignments.

---

## 8. Success Criteria

### Primary Success Criteria

| Criterion | Definition |
|-----------|------------|
| **Positive Sharpe Ratio** | The MoE strategy should achieve a Sharpe ratio > 0 over the out-of-sample period. |
| **Regime Interpretability** | Learned regimes should exhibit distinct return and volatility characteristics, suggesting economic meaning. |
| **Methodological Transparency** | The code, data pipeline, and evaluation framework should be fully reproducible and open-source. |

### Secondary Success Criteria

| Criterion | Definition |
|-----------|------------|
| **Transaction Cost Robustness** | Performance should remain positive after 10 bps transaction costs. |
| **Win Rate > 50%** | The strategy should demonstrate positive returns in a majority of months. |
| **Feature Redundancy Check** | The incremental value of FRED features over VIX-only should be evaluated. |

---

## 9. Assumptions and Constraints

### Methodological Assumptions

1. **Future regimes resemble past regimes** in structure (if not in precise timing).
2. **Macroeconomic indicators** capture relevant, non-redundant regime information.
3. **Monthly rebalancing** is practical and cost-effective for institutional investors.
4. **10 bps transaction costs** are realistic for liquid, large-cap ETFs.

### Data Constraints

1. **Limited sample size** (156 months) limits the complexity of models and prevents deep learning approaches.
2. **US-only data** means findings may not generalize to international markets.
3. **ETF proxies** may not perfectly isolate pure factor exposures.
4. **Macro data revisions** (vintage effects) are not modeled, which is a standard limitation in this type of backtest.

### Practical Constraints

1. **Long-only allocation** only (no short-selling).
2. **Monthly frequency** only (not daily or intraday).
3. **No leverage** (1x exposure only).
4. **Feature dimensionality trade-off:** The use of 96 FRED features requires a sufficiently long training window (132 months) to maintain model stability.

---

## 10. Deliverables

| Deliverable | Description | Format |
|-------------|-------------|--------|
| **Data Pipeline** | End-to-end data loading and preprocessing | Python module |
| **Model Implementations** | All models with a consistent API | Python classes |
| **Backtesting Framework** | Expanding window cross-validation | Python module |
| **Evaluation Module** | Predictive + investment metrics | Python module |
| **Visualization Suite** | Regime plots, performance charts | Python module + PNG |
| **Results Report** | Summary of findings | CSV + Markdown |
| **README** | Project overview and usage | Markdown |

---

## 11. Open-Source Contribution

### Value to the Community

1. **Benchmark:** A transparent, reproducible comparison of multiple approaches on real-world data.
2. **Implementation:** A clean, modular implementation of probabilistic forecasting with mixture density outputs.
3. **Interpretability:** A framework for identifying and analyzing latent economic regimes.
4. **Educational Value:** A practical, well-documented project for practitioners learning factor timing.

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
| 2.0 | 2026-07-29 | Added evaluation framework, success criteria, experimental design |
| 3.0 | 2026-07-30 | Refined scope from "regime-switching" to "regime-aware"; added robustness checks |
| 4.0 | 2026-07-31 | Reformulated research question to emphasize methodological contribution over outperformance; clarified experimental design and constraints |

---

## 14. License

This project is released under the [MIT License](../LICENSE).

---

*For questions or contributions, please open an issue or pull request on GitHub.*
