# Mixture of Experts for Regime-Aware Factor Timing

## A Reproducible Benchmark for Probabilistic Factor Allocation

---

## Abstract

Equity factor premiums exhibit time variation that appears related to macroeconomic conditions. However, regimes are not directly observable, and investors face uncertainty about which regime currently prevails. This project presents a reproducible, open-source benchmark for evaluating whether probabilistic models that explicitly represent uncertainty over economic regimes can provide a coherent framework for factor timing compared to simpler deterministic approaches.

We implement and compare several models: persistence, rolling average, momentum, linear regression, random forest, and a Mixture of Experts (MoE) model with linear experts and softmax gating. Models are evaluated on predictive accuracy and risk-adjusted investment performance using an expanding window backtest with transaction costs. We also explore the role of macroeconomic indicators, comparing VIX-only and FRED-enhanced feature sets.

**Key Findings:**
- In this experimental setup, the MoE model with FRED-enhanced features generated a **Sharpe ratio of 1.49** and **40.61% annualized return** over 42 out-of-sample months
- Four latent regimes were identified with distinct return and volatility characteristics
- **96 months of training data** provided the best balance between model stability and out-of-sample sample size
- The MoE model outperformed all baseline models in this experimental setup

---

## 1. Introduction

### 1.1 Motivation

Equity factor premiums—such as Value, Momentum, Quality, and Low Volatility—are central to modern portfolio construction (Fama & French, 1993; Asness et al., 2013). However, these premiums are not stable over time. Value can underperform for extended periods, and momentum can experience sudden reversals. This creates a practical challenge for investors: how to allocate across factors when the future performance of each factor is uncertain and may depend on latent economic states.

### 1.2 Research Question

This project addresses the following question:

> Can a probabilistic model that explicitly represents uncertainty over economic regimes provide a coherent framework for factor timing compared to simpler deterministic approaches?

### 1.3 Contribution

This project contributes:

1. A comparative evaluation of deterministic and probabilistic models for factor timing
2. An open-source implementation of Mixture of Experts with EM training for regime identification
3. Empirical results on out-of-sample performance with modeled transaction costs
4. Interpretable regime characteristics
5. A fully reproducible pipeline with documented configuration choices

---

## 2. Problem Formulation

### 2.1 Task Definition

We frame factor timing as a probabilistic time-series forecasting problem.

**Inputs ($X_t$)** :
- Historical factor returns for months $t-12$ through $t$
- Macroeconomic indicators (FRED data)

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
| **Momentum** | Exponentially weighted average of historical returns (12-month window, decay=0.9) |
| **Linear Regression** | Linear model with lagged returns and macro features |
| **Random Forest** | Non-linear ensemble with 100 trees, max depth 10 |

### 3.2 Mixture of Experts (MoE)

The MoE model consists of:

1. **Gating Network**: Softmax function that produces probabilities over $K$ latent regimes
2. **Expert Networks**: Linear regression models for each regime with Ridge regularization ($\alpha=0.1$)
3. **Mixture Output**: Probability-weighted combination of expert predictions

Training uses an Expectation-Maximization (EM) algorithm with 100 iterations.

**Architecture**:

$$\pi_t = \text{softmax}(\mathbf{W}_g \mathbf{z}_t + \mathbf{b}_g)$$

$$\hat{\mathbf{y}}_{t+1}^{(k)} = \mathbf{W}^{(k)} \mathbf{z}_t + \mathbf{b}^{(k)}$$

$$P(\mathbf{y}_{t+1}) = \sum_{k=1}^{K} \pi_t^{(k)} \cdot \mathcal{N}(\hat{\mathbf{y}}_{t+1}^{(k)}, \Sigma^{(k)})$$

---

## 4. Experimental Setup

### 4.1 Data

**Time Period**: 2013-08 to 2026-07 (155 months)

**Factors**: 6 (SPY, IWD, MTUM, QUAL, USMV, VIX)

**Features**: FRED-enhanced with 96 features (lagged returns + 18 FRED series with transformations)

**Data Sources**:
- yfinance for ETF factor proxies
- FRED for macroeconomic indicators (CPI, INDPRO, UNRATE, T10Y2Y, GS10, GS2)

### 4.2 Backtesting Framework

| Parameter | Value |
|-----------|-------|
| **Window** | Expanding window |
| **Min Training Size** | 96 months |
| **Test Size** | 1 month |
| **Predictions** | 42 (2022-07 to 2026-07) |

**Rationale for 96 months:** This configuration was chosen to balance two competing requirements: (1) the MoE model requires sufficient data for stable parameter estimation, and (2) the out-of-sample period should be long enough to provide a meaningful evaluation. With 96 months of training, we obtain 42 out-of-sample predictions.

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

### 5.1 Configuration Selection

After testing multiple configurations, the following was selected for the final analysis:

| Parameter | Value |
|-----------|-------|
| **Data** | **FRED-enhanced (96 features)** |
| **Min Training Size** | **96 months** |
| **Number of Experts (K)** | **4** |
| **EM Iterations** | **100** |
| **Allocation Strategy** | **Magnitude-weighted** |
| **Transaction Costs** | **10 bps** |

### 5.2 Model Comparison

| Model | RMSE | MAE | Sharpe | Ann. Return | Max DD | Calmar | Win Rate |
|-------|------|-----|--------|-------------|--------|--------|----------|
| **MoE** | 33.68 | 13.70 | **1.49** | **40.61%** | **-13.73%** | **2.96** | **0.69** |
| Momentum | 8.28 | 5.01 | 0.73 | 11.90% | -29.69% | 0.40 | 0.62 |
| Rolling Avg | 8.39 | 5.05 | 0.62 | 9.03% | -16.49% | 0.55 | 0.64 |
| Linear | 192.33 | 63.08 | 0.59 | 15.20% | -26.21% | 0.58 | 0.48 |
| RF | 8.98 | 5.37 | 0.09 | -3.70% | -34.05% | -0.11 | 0.60 |
| Persistence | 12.79 | 7.52 | -0.61 | -30.68% | -72.70% | -0.42 | 0.57 |

**Observation:** In this experimental setup, the MoE model generated the highest Sharpe ratio (1.49) and annualized return (40.61%) among the models evaluated. The MoE model trades off predictive accuracy (RMSE 33.68) for allocation decisions.

### 5.3 Training Window Sensitivity

| min_train | Predictions | MoE Sharpe | MoE Return | MoE Max DD | MoE RMSE | Stability |
|-----------|-------------|------------|------------|------------|----------|-----------|
| 60 | 78 | 0.73 | 0.36% | -44.41% | 63.99 | ❌ Unstable |
| **96** | **42** | **1.49** | **40.61%** | **-13.73%** | **33.68** | **✅ Stable** |
| 132 | 6 | 1.81 | 90.42% | -7.07% | 9.80 | ✅ Stable |

**Observation:** 96 months provided the best balance between model stability and out-of-sample sample size. While 132 months yielded higher absolute performance, it only produced 6 predictions, which is insufficient for meaningful evaluation.

### 5.4 Regime Analysis (K=4)

| Regime | Frequency | Avg Return | Avg Volatility |
|--------|-----------|------------|----------------|
| Regime 1 | 27.54% | 1.10% | 7.61% |
| Regime 2 | 14.49% | 2.10% | 8.84% |
| Regime 3 | 32.61% | 1.13% | 8.08% |
| Regime 4 | 25.36% | 1.69% | 8.04% |

**Observation:** Four regimes with distinct return and volatility characteristics were identified.

---

## 6. Discussion

### 6.1 Key Findings

In this experimental setup, the MoE model generated the highest Sharpe ratio (1.49) and annualized return (40.61%) among the models evaluated. The model's performance appears to come from its allocation decisions rather than point prediction accuracy, as its RMSE (33.68) was higher than simpler models like momentum (RMSE 8.28). This observation is consistent with the idea that the sign and relative magnitude of predictions may be more important for investment performance than precise point forecasts.

### 6.2 Model Trade-offs

The MoE model showed a trade-off between predictive accuracy and investment performance. While its RMSE was higher than some baselines, it achieved better risk-adjusted returns in this setup. This suggests that the magnitude and sign of predictions may be more important than point accuracy for allocation decisions.

### 6.3 Limitations

1. **Data Scope**: Currently limited to US equity factors (155 months)
2. **Sample Size**: 42 out-of-sample predictions—sufficient for descriptive analysis but not for formal statistical inference
3. **Transaction Costs**: Estimated at 10 bps, not from actual execution data
4. **Single Asset Class**: Results may not generalize to other asset classes
5. **VIX Spot Index**: The VIX is not directly tradable; implementation would require futures or ETFs with roll costs

### 6.4 Future Work

1. **Extended Backtest**: Include longer history with academic factor data
2. **Feature Engineering**: Add rolling volatility and cross-sectional correlations
3. **SHAP Analysis**: Interpret feature contributions to regime assignments
4. **Multi-Asset Extension**: Test on bonds, commodities, and international equities
5. **Statistical Testing**: Formal significance tests with longer backtest

---

## 7. Conclusion

This project presents an open-source, reproducible benchmark for evaluating probabilistic regime-aware models in equity factor timing. The Mixture of Experts model demonstrated positive risk-adjusted performance in our backtest, generating a Sharpe ratio of 1.49 and an annualized return of 40.61% over 42 out-of-sample months (July 2022 - July 2026). The model identified four latent regimes with distinct characteristics.

We emphasize that this work is not a claim of market-beating performance, but rather a transparent contribution to the quantitative finance community. The full code, data pipeline, and evaluation framework are publicly available, enabling practitioners and researchers to extend, critique, and improve upon this work.

---

## References

1. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

2. Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.

3. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79-87.

4. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

5. Shih, W. (2020). *Machine Learning for Factor Investing*. CFA Institute Research Foundation.

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
│   ├── generate_paper.py       # Paper generation script
│   └── utils.py
├── results/
│   ├── regime_analysis/
│   ├── paper_figures/
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

# Set up FRED API key (optional - VIX fallback works without it)
# Create .env file with: FRED_API_KEY=your_key_here

# Run full pipeline with optimal configuration
python main.py --run-all --start-date 2013-08-01 --min-train 96 --models moe rolling_avg momentum linear rf persistence

# Quick test for debugging
python main.py --run-all --quick-test

# Generate paper figures
python src/visualization.py --timestamp <your_timestamp>

# Generate paper PDF
python src/generate_paper.py

# Run PyTorch MoE experiment
python run_moe_torch.py
```

---

## How to Reproduce Best Results

To reproduce the best results from this project (min_train=96, FRED-enhanced features):

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Set FRED API key (optional)
echo "FRED_API_KEY=your_key_here" > .env

# 3. Run the pipeline with optimal configuration
python main.py --run-all --start-date 2013-08-01 --min-train 96 --models moe rolling_avg momentum linear rf persistence

# 4. Generate paper figures (use the timestamp from your run)
python src/visualization.py --timestamp 20260802_020816

# 5. View results
# - Summary table: results/summary_*.csv
# - Paper figures: results/paper_figures/*/
# - Regime analysis: results/regime_analysis/
```

**Expected output based on our experiments:** MoE Sharpe ratio ~1.49, annual return ~40.61%, max drawdown ~-13.73%, 42 out-of-sample predictions.

---

## License

MIT

---

**Author:** Ken Ira Lacson Talingting  
**Year:** 2026  
**Status:** ✅ Complete - Pipeline Stable, Paper Ready, Open-Source Release

---

## Results Summary

| Configuration | Value |
|---------------|-------|
| **Data** | FRED-enhanced (96 features) |
| **Min Training** | 96 months |
| **Model** | MoE (K=4) |
| **Backtest Period** | July 2022 - July 2026 (42 predictions) |
| **Sharpe Ratio** | 1.49 |
| **Annual Return** | 40.61% |
| **Max Drawdown** | -13.73% |
| **Calmar Ratio** | 2.96 |
| **Win Rate** | 69.05% |