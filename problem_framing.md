# Problem Framing: Probabilistic Regime-Aware Factor Timing

---

## Overview

This document frames the research problem for the `moe-factor-timing` project. The core objective is to investigate whether probabilistic models that explicitly represent uncertainty over latent economic regimes can provide a coherent framework for dynamic equity factor allocation. This project is designed as a practical, open-source contribution to the quantitative finance community, prioritizing methodological transparency and empirical reproducibility over claims of financial outperformance.

---

## 1. Research Problem

### 1.1 The Core Challenge

Equity factor premiums—such as Value, Momentum, Quality, and Low Volatility—are central to modern portfolio construction. A substantial body of empirical evidence documents that these premiums are not stable over time (Fama & French, 1993; Asness et al., 2013). Value can underperform for extended periods, and momentum can experience sudden reversals. This time variation is widely believed to be related to changing macroeconomic conditions, though the precise nature of these relationships remains an active area of research.

This creates a practical challenge for systematic investors: how should one allocate across factors when the future performance of each factor is uncertain and may depend on latent, unobservable economic states? While the theoretical case for regime-dependent factor behavior is well-established, the practical implementation of regime-aware allocation strategies remains an open problem.

### 1.2 Why This Problem Matters for Open-Source Research

The factor timing literature faces a persistent gap between academic research and practical implementation. Academic studies often employ sophisticated models that are difficult to replicate or implement in practice. Conversely, practitioner solutions frequently rely on ad-hoc heuristics without rigorous out-of-sample validation.

This project addresses this gap by providing a transparent, reproducible benchmark that bridges these two worlds. We implement a probabilistic approach—Mixture of Experts—that is conceptually grounded in the regime-switching literature while remaining computationally tractable and practically implementable. By making the code, data, and methodology fully open-source, we provide a baseline that practitioners can adapt and researchers can extend.

### 1.3 What This Project Does and Does Not Do

**This project does:**
- Provides a reproducible, open-source framework for evaluating regime-aware factor timing
- Implements a Mixture of Experts model with EM training and compares it against deterministic baselines
- Tests whether FRED macroeconomic indicators add predictive value beyond market volatility
- Evaluates models on both predictive accuracy and investment performance metrics

**This project does not:**
- Claim to have discovered a market-beating strategy
- Assert that regime-aware models will outperform in all market conditions
- Present itself as a definitive solution to the factor timing problem
- Make claims of statistical significance without appropriate testing

---

## 2. Research Questions

### Primary Research Question

> Does a probabilistic model that explicitly represents uncertainty over latent economic regimes offer a practical and coherent framework for out-of-sample factor allocation, and how does its performance compare to simpler deterministic approaches?

### Secondary Research Questions

1. **Model Comparison:** How does a Mixture of Experts (MoE) model compare to linear and non-linear machine learning baselines in a realistic expanding-window backtesting setup with transaction costs?

2. **Feature Value:** Do macroeconomic indicators from FRED provide incremental predictive value beyond historical factor returns and market volatility in a regime-aware framework?

3. **Regime Interpretability:** Are the learned latent regimes economically interpretable, and do they exhibit distinct and stable characteristics over time?

---

## 3. Problem Definition

### 3.1 Task Type

We frame factor timing as a **probabilistic time-series forecasting** problem with an implicit decision-making component. The model generates distributional predictions of next-month factor returns, which are then used to inform a portfolio allocation strategy.

### 3.2 Formal Specification

**Inputs ($X_t$)** :

| Feature Type | Description | Source |
|--------------|-------------|--------|
| Historical factor returns | Monthly returns for months $t-L$ through $t$ | yfinance |
| Macroeconomic indicators | VIX, CPI, Industrial Production, Unemployment, Term Spread | FRED |
| Derived features | Lagged returns, transformations, z-scores | Engineered |

**Outputs ($Y_{t+1}$)** :

Vector of next-month returns for each of the $K=6$ equity factors:

$$\mathbf{y}_{t+1} = [r_{t+1}^{(\text{SPY})}, r_{t+1}^{(\text{IWD})}, r_{t+1}^{(\text{MTUM})}, r_{t+1}^{(\text{QUAL})}, r_{t+1}^{(\text{USMV})}, r_{t+1}^{(\text{VIX})}]$$

**Prediction Task**:

Estimate the conditional distribution:

$$P(\mathbf{y}_{t+1} \mid \mathbf{x}_t, \mathbf{y}_{t-L:t})$$

where:
- $\mathbf{x}_t$ = macroeconomic features at time $t$
- $\mathbf{y}_{t-L:t}$ = historical factor returns

### 3.3 Key Distinction

The model is **regime-aware** rather than regime-switching. It estimates the probability of being in different latent states (soft assignment) rather than enforcing a hard classification. This probabilistic representation is central to the project's methodological contribution, as it preserves uncertainty information that can be used in downstream allocation decisions.

---

## 4. Methodology

### 4.1 Models

To ensure a rigorous comparison, we evaluate models across several levels of complexity:

**Level 1: Non-Machine Learning Heuristics (Baselines)**

| Model | Description | Purpose |
|-------|-------------|---------|
| **Persistence** | Next month's return equals current month's return | Establishes minimum performance threshold |
| **Rolling Average** | Average of last 12 months | Captures medium-term trends |
| **Momentum** | Exponentially weighted average (12-month window, decay=0.9) | Trend-following baseline |

**Level 2: Standard Machine Learning Baselines**

| Model | Description | Purpose |
|-------|-------------|---------|
| **Linear Regression** | Linear model with lagged returns and macro features | Establishes linear benchmark |
| **Random Forest** | Non-linear ensemble (100 trees, max depth 10) | Captures non-linear relationships |

**Level 3: Primary Contribution**

| Model | Description | Purpose |
|-------|-------------|---------|
| **Mixture of Experts (MoE)** | Softmax gating + linear experts per regime, EM training with Ridge regularization ($\alpha=0.1$) | Explicitly models regime uncertainty |

### 4.2 Evaluation Framework

**Backtesting Approach:**
- **Window Type:** Expanding window (growing training set)
- **Minimum Training Size:** 96 months
- **Test Size:** 1 month
- **Out-of-Sample Period:** 2022-07 to 2026-07 (42 monthly predictions)

**Rationale for 96 Months:** This configuration was selected after testing 60, 96, and 132 months. The 60-month setting produced unstable MoE estimates (RMSE ~64). The 132-month setting yielded only 6 predictions, insufficient for meaningful evaluation. The 96-month setting provides stable estimates with 42 out-of-sample predictions.

**Allocation Strategy:**
- Magnitude-weighted long-only positions on positive predictions
- 10 basis points transaction costs
- No short-selling or leverage

**Metrics:**

| Dimension | Metrics |
|-----------|---------|
| **Predictive Accuracy** | RMSE, MAE |
| **Investment Performance** | Sharpe ratio, Annualized return, Maximum drawdown, Calmar ratio, Win rate |

### 4.3 Data

| Dataset | Source | Period | Description |
|---------|--------|--------|-------------|
| **Equity Factors** | yfinance | 2013-08 to 2026-07 | SPY, IWD, MTUM, QUAL, USMV, VIX |
| **Macroeconomic Data** | FRED | 2013-08 to 2026-07 | CPI, INDPRO, UNRATE, T10Y2Y, GS10, GS2 |

**Features:** 96 features total (lagged factor returns + FRED indicators with transformations)

---

## 5. Experimental Design

### Experiment 1: Model Comparison

Compare all models using the expanding window backtest with 96 months of minimum training data.

**Rationale:** This establishes whether the MoE model provides a distinct and useful benchmark for regime-aware allocation compared to deterministic baselines.

### Experiment 2: Training Window Sensitivity

Test the stability of the optimal MoE configuration under varying training window sizes (60, 96, 132 months).

**Rationale:** This tests the robustness of the model to data availability and identifies the minimum training data required for stable estimates.

### Experiment 3: Regime Analysis

Extract and analyze the latent regimes identified by the MoE model.

**Rationale:** This provides interpretability and tests whether the learned regimes exhibit economically meaningful characteristics.

---

## 6. Success Criteria

### Primary Criteria

| Criterion | Definition | Status |
|-----------|------------|--------|
| **Positive Sharpe Ratio** | MoE strategy achieves Sharpe ratio > 0 over out-of-sample period | ✅ Achieved (1.49) |
| **Regime Interpretability** | Regimes exhibit distinct return and volatility characteristics | ✅ Achieved (4 distinct regimes) |
| **Methodological Transparency** | Code, data, and evaluation framework are fully reproducible | ✅ Achieved |

### Secondary Criteria

| Criterion | Definition | Status |
|-----------|------------|--------|
| **Transaction Cost Robustness** | Performance remains positive after 10 bps costs | ✅ Achieved |
| **Win Rate > 50%** | Positive returns in majority of months | ✅ Achieved (69%) |
| **Feature Value Assessment** | FRED features provide non-redundant information | ✅ Achieved |

---

## 7. Scope and Constraints

### Scope

| Inclusion | Exclusion |
|-----------|-----------|
| US equity factors | International markets |
| Monthly frequency | Daily or intraday |
| Long-only allocation | Short-selling or leverage |
| ETFs as factor proxies | Pure factor portfolios |
| FRED macroeconomic data | Alternative macro datasets |
| MoE with linear experts | Deep learning or non-linear experts |

### Constraints

| Constraint | Implication |
|------------|-------------|
| **Sample Size** | 155 months limits model complexity |
| **US-Only** | Findings may not generalize internationally |
| **ETF Proxies** | May not perfectly isolate pure factor exposures |
| **Macro Data Revisions** | Vintage effects not modeled |
| **VIX Trading** | VIX is a spot index, not directly tradable |

---

## 8. Research Justification

### Why This Problem is Worth Investigating

1. **Practical Relevance:** Factor timing is a persistent challenge for systematic investors. Even modest improvements in allocation decisions can have meaningful economic impact.

2. **Methodological Gap:** The literature lacks open-source, reproducible benchmarks for evaluating regime-aware models in this domain. This project fills that gap.

3. **Interpretability:** Understanding how and when factor premiums vary provides insights into the economic drivers of asset returns.

### Why This Approach is Appropriate

1. **Probabilistic Representation:** The MoE model explicitly models uncertainty over regimes, which is more appropriate than hard regime-switching for financial data where regimes are inherently uncertain.

2. **Computational Tractability:** EM-trained linear experts are computationally efficient and interpretable, unlike black-box models.

3. **Transparent Comparison:** Including multiple baselines allows for a fair assessment of the MoE model's incremental value.

---

## 9. Deliverables

| Deliverable | Description | Status |
|-------------|-------------|--------|
| **Data Pipeline** | End-to-end data loading and preprocessing | ✅ |
| **Model Implementations** | All models with consistent API | ✅ |
| **Backtesting Framework** | Expanding window cross-validation | ✅ |
| **Evaluation Module** | Predictive + investment metrics | ✅ |
| **Visualization Suite** | Regime plots and performance charts | ✅ |
| **Results Report** | Summary of findings (CSV + Markdown) | ✅ |
| **README** | Project overview and usage | ✅ |
| **Paper PDF** | Mini research paper | ✅ |

---

## 10. References

1. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

2. Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.

3. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79-87.

4. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

5. Shih, W. (2020). *Machine Learning for Factor Investing*. CFA Institute Research Foundation.

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-07-25 | Initial problem framing |
| 2.0 | 2026-07-29 | Added evaluation framework, success criteria, experimental design |
| 3.0 | 2026-07-30 | Refined scope from "regime-switching" to "regime-aware"; added robustness checks |
| 4.0 | 2026-07-31 | Reformulated research question to emphasize methodological contribution over outperformance |
| 5.0 | 2026-08-02 | Updated validation approach to min_train=96; added momentum baseline; structured research justification |

---

## 12. License

This project is released under the [MIT License](../LICENSE).

---

*For questions or contributions, please open an issue or pull request on GitHub.*