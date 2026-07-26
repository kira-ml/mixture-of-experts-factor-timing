# Mixture of Experts for Regime-Switching Factor Timing

## An Empirical Investigation of Probabilistic Regime Models for Equity Factor Allocation

---

## Abstract

Equity factor premiums exhibit significant time variation that appears related to macroeconomic conditions. However, regimes are not directly observable, and investors face uncertainty about which regime currently prevails. This project investigates whether probabilistic models that explicitly represent uncertainty over economic regimes can improve out-of-sample factor timing performance compared to simpler deterministic approaches.

We implement and compare several models: persistence, rolling average, linear regression, random forest, and a Mixture of Experts (MoE) model with linear experts and softmax gating. We also develop an isolated PyTorch-based MoE with LSTM gating and MLP experts for comparison. Models are evaluated on predictive accuracy and risk-adjusted investment performance using an expanding window backtest with transaction costs.

Preliminary results suggest that the SimpleMoE model achieves a Sharpe ratio of 0.506 with a 13.94% annualized return in the out-of-sample period (2019–2026). The PyTorch MoE shows improved predictive accuracy (RMSE: 14.67 vs 15.54) but lower investment performance (Sharpe: 0.153). Four latent regimes are identified with distinct return and volatility characteristics. Work is ongoing to incorporate macroeconomic indicators from FRED.

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
3. An isolated PyTorch implementation with LSTM gating and MLP experts
4. Empirical results on out-of-sample performance with realistic transaction costs
5. Interpretable regime characteristics mapped to economic conditions

---

## 2. Problem Formulation

### 2.1 Task Definition

We frame factor timing as a probabilistic time-series forecasting problem.

**Inputs ($X_t$)** :
- Historical factor returns for months $t-12$ through $t$
- Macroeconomic indicators (VIX; FRED data in progress)

**Outputs ($Y_{t+1}$)** :
- Vector of next-month returns for each of the $K$ equity factors

**Prediction Task**:
Estimate the conditional distribution:

$$P(\mathbf{y}_{t+1} \mid \mathbf{x}_t, \mathbf{y}_{t-L:t})$$

where $\mathbf{x}_t$ represents macroeconomic features and $\mathbf{y}_{t-L:t}$ represents historical factor returns.

### 2.2 Factors

| Factor | Proxy | Description |
|--------|-------|-------------|
| Market | SPY | S&P 500 |
| Value | IWD | Russell 1000 Value |
| Momentum | MTUM | MSCI USA Momentum |
| Quality | QUAL | MSCI USA Quality |
| Low Volatility | USMV | MSCI USA Min Volatility |
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

### 3.2 Mixture of Experts (SimpleMoE)

The SimpleMoE model consists of:

1. **Gating Network**: Softmax function that produces probabilities over $K$ latent regimes
2. **Expert Networks**: Linear regression models for each regime
3. **Mixture Output**: Probability-weighted combination of expert predictions

Training uses an Expectation-Maximization (EM) algorithm with 100 iterations.

**Architecture**:

$$\pi_t = \text{softmax}(\mathbf{W}_g \mathbf{z}_t + \mathbf{b}_g)$$

$$\hat{\mathbf{y}}_{t+1}^{(k)} = \mathbf{W}^{(k)} \mathbf{z}_t + \mathbf{b}^{(k)}$$

$$P(\mathbf{y}_{t+1}) = \sum_{k=1}^{K} \pi_t^{(k)} \cdot \mathcal{N}(\hat{\mathbf{y}}_{t+1}^{(k)}, \Sigma^{(k)})$$

### 3.3 PyTorch MoE (Isolated Experiment)

An alternative implementation using:

- **Gating Network**: LSTM that processes 12-month sequences (hidden size: 32)
- **Expert Networks**: MLPs with 1 hidden layer (32 neurons)
- **Training**: Gradient descent with early stopping (patience: 20)

This implementation is isolated from the main pipeline for safe experimentation.

---

## 4. Experimental Setup

### 4.1 Data

**Time Period**: 2013-08 to 2026-07 (156 months)

**Factors**: 6 (SPY, IWD, MTUM, QUAL, USMV, VIX)

**Features**: 28 (lagged returns + VIX)

**Macro Data**: FRED integration in progress (CPI, INDPRO, UNRATE, T10Y2Y)

### 4.2 Backtesting Framework

| Parameter | Value |
|-----------|-------|
| **Window** | Expanding window |
| **Min Training Size** | 60 months |
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

### 5.1 Model Comparison

#### SimpleMoE (Recommended Model)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.5060 |
| Annual Return | 13.94% |
| Volatility | 46.86% |
| Max Drawdown | -52.29% |
| Calmar Ratio | 0.2666 |
| Win Rate | 57.83% |
| RMSE | 15.54 |
| MAE | 8.50 |

#### PyTorch MoE (Expanding Window)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.1528 |
| Annual Return | -6.20% |
| Volatility | 56.19% |
| Max Drawdown | -72.03% |
| RMSE | 14.67 |
| MAE | 8.14 |

#### Comparison

| Aspect | SimpleMoE | PyTorch MoE |
|--------|-----------|-------------|
| Investment Performance | Higher (Sharpe 0.506) | Lower (Sharpe 0.153) |
| Predictive Accuracy | Lower (RMSE 15.54) | Higher (RMSE 14.67) |

The SimpleMoE demonstrates better risk-adjusted performance in the out-of-sample period, despite slightly lower predictive accuracy.

### 5.2 Regime Analysis (K=4)

| Regime | Frequency | Avg Return | Avg Volatility |
|--------|-----------|------------|----------------|
| Regime 1 | 38.46% | 0.85% | 8.64% |
| Regime 2 | 18.88% | 1.25% | 8.38% |
| Regime 3 | 18.88% | 1.15% | 5.48% |
| Regime 4 | 23.78% | 2.79% | 8.01% |

Regime 4 shows the highest average return (2.79%), while Regime 3 exhibits the lowest volatility (5.48%). These differences suggest economically meaningful regime separation.

### 5.3 Hyperparameter Sensitivity

| Parameter | Values Tested | Optimal |
|-----------|---------------|---------|
| Number of Experts (K) | 2, 3, 4, 5 | K=4 |
| EM Iterations | 30, 100 | 100 |
| Allocation Strategy | Equal-weight, Magnitude-weighted | Magnitude-weighted |
| Transaction Costs | 0, 10 bps | 10 bps (negligible impact) |

---

## 6. Discussion

### 6.1 Interpretation

The identification of four distinct regimes with different return and volatility characteristics provides some evidence that latent economic regimes exist and can be learned from factor return data. Regime 4, which occurs approximately 24% of the time, is associated with the highest average returns and appears to represent a favorable environment for factor investing.

### 6.2 Model Trade-offs

The SimpleMoE model demonstrates a trade-off between predictive accuracy and investment performance. While its RMSE (15.54) is higher than the PyTorch MoE (14.67), it achieves better risk-adjusted returns. This suggests that the magnitude and sign of predictions matter more than point accuracy for allocation decisions.

### 6.3 Limitations

1. **Data Scope**: Currently limited to US equity factors and one macro proxy (VIX)
2. **Sample Size**: 156 monthly observations limits model complexity
3. **Transaction Costs**: Estimated, not from actual execution data
4. **Model Complexity**: SimpleMoE uses linear experts; non-linear relationships may be missed
5. **Single Asset Class**: Results may not generalize to other asset classes

### 6.4 Future Work

1. **Macroeconomic Data**: Integrate FRED indicators (CPI, INDPRO, UNRATE, T10Y2Y)
2. **HMM Implementation**: Compare with Hidden Markov Model for regime detection
3. **Feature Engineering**: Add rolling volatility and cross-sectional correlations
4. **Extended Backtest**: Include longer history with academic factor data
5. **SHAP Analysis**: Interpret feature contributions to regime assignments

---

## 7. Conclusion

This project presents an empirical investigation of probabilistic models for factor timing. The SimpleMoE model, with 4 experts, 100 EM iterations, and magnitude-weighted allocation, achieves a Sharpe ratio of 0.506 and 13.94% annualized return in the out-of-sample period. Four latent regimes are identified with distinct return and volatility characteristics.

The PyTorch MoE implementation shows improved predictive accuracy but lower investment performance, suggesting that additional complexity does not necessarily translate to better allocation decisions with the current data.

These results provide preliminary evidence that probabilistic regime models may offer a useful framework for factor timing, though further validation with macroeconomic data and longer histories is needed.

---

## References

1. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

2. Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.

3. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79-87.

4. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

5. Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68.

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

# Run full pipeline
python main.py --run-all

# Run PyTorch MoE experiment
python run_moe_torch.py
```

---

## License

MIT

---

**Author:** Ken Ira L. Alacson  
**Year:** 2026  
**Status:** Active Development