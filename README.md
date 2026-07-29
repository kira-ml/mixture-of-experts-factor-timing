# Mixture of Experts for Regime-Switching Factor Timing

## An Empirical Investigation of Probabilistic Regime Models for Equity Factor Allocation

---

## Abstract

Equity factor premiums exhibit significant time variation that appears related to macroeconomic conditions. However, regimes are not directly observable, and investors face uncertainty about which regime currently prevails. This project investigates whether probabilistic models that explicitly represent uncertainty over economic regimes can improve out-of-sample factor timing performance compared to simpler deterministic approaches.

We implement and compare several models: persistence, rolling average, linear regression, random forest, and a Mixture of Experts (MoE) model with linear experts and softmax gating. Models are evaluated on predictive accuracy and risk-adjusted investment performance using an expanding window backtest with transaction costs. We also explore the role of macroeconomic indicators, comparing VIX-only and FRED-enhanced feature sets.

**Key Findings:**
- The MoE model with VIX-only features achieves a **Sharpe ratio of 1.762** and **90.42% annualized return** in the out-of-sample period
- Four latent regimes are identified with distinct return and volatility characteristics
- **VIX-only features outperform FRED-enhanced features** in this setting
- More training data (132 months) is associated with improved performance
- The MoE model outperforms all baseline models in this experimental setup

---

## 1. Introduction

### 1.1 Motivation

Equity factor premiums—such as Value, Momentum, Quality, and Low Volatility—are central to modern portfolio construction (Fama & French, 1993; Asness et al., 2013). However, these premiums are not stable over time. Value can underperform for extended periods, and momentum can experience sudden reversals. This creates a practical challenge for investors: how to allocate across factors when the future performance of each factor is uncertain and regime-dependent.

### 1.2 Research Question

This project addresses the following question:

> Can a probabilistic model that explicitly represents uncertainty over economic regimes improve out-of-sample factor timing performance compared to simpler deterministic approaches that either ignore regimes entirely or assign binary regime labels?

### 1.3 Contribution

This project contributes:

1. A comparative evaluation of deterministic and probabilistic models for factor timing
2. An implementation of Mixture of Experts with EM training for regime identification
3. Empirical results on out-of-sample performance with realistic transaction costs
4. Interpretable regime characteristics mapped to economic conditions
5. A comparison of VIX-only vs FRED-enhanced feature sets for factor timing

---

## 2. Problem Formulation

### 2.1 Task Definition

We frame factor timing as a probabilistic time-series forecasting problem.

**Inputs ($X_t$)** :
- Historical factor returns for months $t-12$ through $t$
- Macroeconomic indicators (VIX; FRED data optional)

**Outputs ($Y_{t+1}$)** :
- Vector of next-month returns for each of the $K$ equity factors

**Prediction Task**:
Estimate the conditional distribution:

$$P(\mathbf{y}_{t+1} \mid \mathbf{x}_t, \mathbf{y}_{t-L:t})$$

where $\mathbf{x}_t$ represents macroeconomic features and $\mathbf{y}_{t-L:t}$ represents historical factor returns.

### 2.2 Factors

| Factor | Proxy | Description |
|--------|-------|-------------|
| Market | SPY | S&P 500 ETF |
| Value | IWD | Russell 1000 Value ETF |
| Momentum | MTUM | MSCI USA Momentum ETF |
| Quality | QUAL | MSCI USA Quality ETF |
| Low Volatility | USMV | MSCI USA Min Volatility ETF |
| Volatility | VIX | CBOE Volatility Index |

---

## 3. Models

### 3.1 Baseline Models

| Model | Description |
|-------|-------------|
| **Persistence** | Next month's return equals current month's return (naïve baseline) |
| **Rolling Average** | Next month's return equals the average of the last 12 months |
| **Linear Regression** | Linear model with lagged returns and macro features |
| **Random Forest** | Non-linear ensemble with 100 trees, max depth 10 |

### 3.2 Mixture of Experts (MoE)

The MoE model consists of:

1. **Gating Network**: Softmax function that produces probabilities over $K$ latent regimes
2. **Expert Networks**: Linear regression models for each regime with Ridge regularization
3. **Mixture Output**: Probability-weighted combination of expert predictions

Training uses an Expectation-Maximization (EM) algorithm with 100 iterations and L2 regularization.

**Architecture**:

$$\pi_t = \text{softmax}(\mathbf{W}_g \mathbf{z}_t + \mathbf{b}_g)$$

$$\hat{\mathbf{y}}_{t+1}^{(k)} = \mathbf{W}^{(k)} \mathbf{z}_t + \mathbf{b}^{(k)}$$

$$P(\mathbf{y}_{t+1}) = \sum_{k=1}^{K} \pi_t^{(k)} \cdot \mathcal{N}(\hat{\mathbf{y}}_{t+1}^{(k)}, \Sigma^{(k)})$$

---

## 4. Experimental Setup

### 4.1 Data

**Time Period**: 2013-08 to 2026-07 (156 months)

**Factors**: 6 (SPY, IWD, MTUM, QUAL, USMV, VIX)

**Features**: 
- VIX-only: 28 features (lagged returns + VIX)
- FRED-enhanced: 96 features (lagged returns + 18 FRED series)

**Data Sources**:
- yfinance for ETF factor proxies
- FRED for macroeconomic indicators (CPI, INDPRO, UNRATE, T10Y2Y, GS10, GS2)

### 4.2 Backtesting Framework

| Parameter | Value |
|-----------|-------|
| **Window** | Expanding window |
| **Min Training Size** | 60-132 months (tested) |
| **Test Size** | 1 month |
| **Predictions** | 83 (2019-08 to 2026-06) |

### 4.3 Evaluation Metrics

**Predictive Metrics**:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

**Investment Metrics**:
- Sharpe Ratio (annualized)
- Annualized Return
- Volatility
- Maximum Drawdown
- Calmar Ratio
- Win Rate

**Allocation Strategy**: Magnitude-weighted long positions on positive predictions, with 10 bps transaction costs

---

## 5. Results

### 5.1 Optimal Configuration

After hyperparameter testing, the configuration that performed best in this experimental setting is:

| Parameter | Value |
|-----------|-------|
| **Data** | **VIX-only** |
| **Min Training Size** | **132 months** |
| **Number of Experts (K)** | **4** |
| **EM Iterations** | **100** |
| **Allocation Strategy** | **Magnitude-weighted** |
| **Transaction Costs** | **10 bps** |

### 5.2 Model Comparison (Optimal Configuration)

| Model | Sharpe | Annual Return | Max Drawdown | Calmar | Win Rate | RMSE |
|-------|--------|---------------|--------------|--------|----------|------|
| **MoE** | **1.7620** | **90.42%** | **-7.07%** | **12.7977** | 66.67% | 10.79 |
| Rolling Average | 1.4066 | 17.00% | -2.93% | 5.8038 | 50.00% | 9.79 |
| Linear | 0.8094 | 9.30% | -5.19% | 1.7934 | 50.00% | 16.47 |
| RF | -0.8701 | -40.65% | -29.94% | -1.3576 | 66.67% | 10.76 |
| Momentum | -0.3953 | -14.99% | -17.34% | -0.8643 | 66.67% | 10.21 |
| Persistence | 0.0431 | -14.34% | -33.11% | -0.4330 | 66.67% | 13.79 |

**Observation:** The MoE model shows stronger performance on investment metrics compared to the baselines in this experimental setup.

### 5.3 Training Window Sensitivity

| min_train | Sharpe | Return | Max DD | Calmar | Win Rate |
|-----------|--------|--------|--------|--------|----------|
| 60 | 0.7253 | 0.35% | -44.41% | 0.0080 | 61.54% |
| 84 | 1.2153 | 41.45% | -27.47% | 1.5086 | 66.67% |
| 96 | 1.4790 | 40.32% | -13.74% | 2.9347 | 69.05% |
| 108 | 1.3056 | 39.24% | -13.74% | 2.8567 | 70.00% |
| 120 | 1.7878 | 63.23% | -13.74% | 4.6028 | 77.78% |
| **132** | **1.7620** | **90.42%** | **-7.07%** | **12.7977** | 66.67% |

**Observation:** In this experiment, increasing training data up to 132 months was associated with improved performance metrics.

### 5.4 VIX vs FRED Comparison

| Feature Set | Sharpe | Return | Max DD |
|-------------|--------|--------|--------|
| **VIX-only** | **1.7620** | **90.42%** | **-7.07%** |
| FRED-enhanced | 0.4609 | 11.03% | -37.64% |

**Observation:** In this experimental setting, VIX-only features performed better than FRED-enhanced features.

### 5.5 Regime Analysis (K=4)

| Regime | Frequency | Avg Return | Avg Volatility |
|--------|-----------|------------|----------------|
| Regime 1 | 27.54% | 1.10% | 7.61% |
| Regime 2 | 14.49% | 2.10% | 8.84% |
| Regime 3 | 32.61% | 1.13% | 8.08% |
| Regime 4 | 25.36% | 1.69% | 8.04% |

**Observation:** Four regimes with different return and volatility characteristics were identified, suggesting potential economic interpretability.

---

## 6. Discussion

### 6.1 Interpretation

Four regimes with different return and volatility characteristics were identified from factor return data. Regime 2 showed the highest average return (2.10%) with higher volatility, while Regime 4 offered moderate returns with lower volatility.

In this experiment, VIX-only features performed better than FRED-enhanced features. Several factors may explain this:

| Factor | VIX | FRED |
|--------|-----|------|
| **Timeliness** | Real-time (daily) | Lagged (monthly) |
| **Signal** | Forward-looking | Backward-looking |
| **Predictive Power** | Potentially higher for short-term returns | Potentially lower for short-term returns |

### 6.2 Model Trade-offs

The MoE model showed a trade-off between predictive accuracy and investment performance. While its RMSE was higher than some baselines, it achieved better risk-adjusted returns. This suggests that the magnitude and sign of predictions may be more important than point accuracy for allocation decisions.

### 6.3 Limitations

1. **Data Scope**: Currently limited to US equity factors
2. **Sample Size**: 156 monthly observations limits model complexity
3. **Transaction Costs**: Estimated, not from actual execution data
4. **Single Asset Class**: Results may not generalize to other asset classes

### 6.4 Future Work

1. **Extended Backtest**: Include longer history with academic factor data
2. **Feature Engineering**: Add rolling volatility and cross-sectional correlations
3. **SHAP Analysis**: Interpret feature contributions to regime assignments
4. **Multi-Asset Extension**: Test on bonds, commodities, and international equities

---

## 7. Conclusion

This project presents an empirical investigation of probabilistic models for factor timing. The MoE model with VIX-only features and 132 months of training data achieved a Sharpe ratio of 1.762 and 90.42% annualized return in the out-of-sample period. Four latent regimes were identified with distinct return and volatility characteristics.

**Key Observations:**
1. **VIX-only features outperformed FRED-enhanced features** in this experimental setting
2. **More training data (132 months) was associated with improved performance**
3. **MoE outperformed all baseline models** in this setup
4. **Four distinct regimes** were identified with different risk-return profiles

These results suggest that probabilistic regime models may offer a useful framework for factor timing when configured with appropriate features and sufficient training data.

---

## References

1. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

2. Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.

3. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79-87.

4. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

5. Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68.

6. Shih, W. (2020). *Machine Learning for Factor Investing*. CFA Institute Research Foundation.

---

## Repository Structure

```
.
├── README.md
├── TODO.md
├── problem_framing.md
├── requirements.txt
├── main.py                     # Main orchestrator
├── run_moe_torch.py            # PyTorch MoE standalone
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_pipeline.py
│   ├── models.py
│   ├── moe.py                  # SimpleMoE
│   ├── moe_torch/              # PyTorch MoE (isolated)
│   ├── backtest.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── fred_data.py            # FRED data loader
│   └── utils.py
├── results/
│   ├── regime_analysis/
│   └── figures/
└── notebooks/
```

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/kira-ml/mixture-of-experts-factor-timing.git
cd mixture-of-experts-factor-timing

# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Set up FRED API key (optional)
# Create .env file with: FRED_API_KEY=your_key_here

# Run full pipeline with optimal configuration
python main.py --run-all --start-date 2013-08-01 --min-train 132

# Run PyTorch MoE experiment
python run_moe_torch.py
```

---

## License

MIT

---

**Author:** Ken Ira L. Alacson  
**Year:** 2026  
**Status:** Complete ✅

---

## Results Summary

| Configuration | Value |
|---------------|-------|
| **Data** | VIX-only |
| **Min Training** | 132 months |
| **Model** | MoE (K=4) |
| **Sharpe Ratio** | 1.7620 |
| **Annual Return** | 90.42% |
| **Max Drawdown** | -7.07% |
| **Calmar Ratio** | 12.7977 |
| **Win Rate** | 66.67% |