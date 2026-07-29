"""
Visualization and analysis for MoE regime probabilities and paper-ready metrics.
Generates plots and exports performance metrics for the mini research paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
import json

from src.utils import get_results_dir, ensure_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# ORIGINAL REGIME ANALYSIS FUNCTIONS (KEPT UNCHANGED)
# ============================================================================

def extract_regime_data(model, X: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Extract regime probabilities from fitted MoE model."""
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    probs = model.predict_proba(X_clean)
    regime_df = pd.DataFrame(
        probs, index=dates, columns=[f'Regime_{i+1}' for i in range(probs.shape[1])]
    )
    regime_df['dominant_regime'] = np.argmax(probs, axis=1) + 1
    regime_df['dominant_prob'] = np.max(probs, axis=1)
    return regime_df


def analyze_regime_characteristics(regime_df: pd.DataFrame, returns_df: pd.DataFrame) -> Dict:
    """Analyze characteristics of each regime."""
    n_regimes = regime_df.shape[1] - 2
    stats = {}
    common_idx = regime_df.index.intersection(returns_df.index)
    
    if len(common_idx) == 0:
        logger.warning("No overlapping indices between regime_df and returns_df")
        for i in range(n_regimes):
            regime_col = f'Regime_{i+1}'
            stats[regime_col] = {'frequency': 0, 'avg_return': np.nan, 'avg_volatility': np.nan, 'avg_prob': 0, 'n_periods': 0}
        return stats
    
    regime_df_aligned = regime_df.loc[common_idx]
    returns_df_aligned = returns_df.loc[common_idx]
    
    for i in range(n_regimes):
        regime_col = f'Regime_{i+1}'
        mask = regime_df_aligned['dominant_regime'] == i + 1
        if mask.sum() == 0:
            stats[regime_col] = {'frequency': 0, 'avg_return': np.nan, 'avg_volatility': np.nan, 'avg_prob': 0, 'n_periods': 0}
            continue
        stats[regime_col] = {
            'frequency': mask.sum() / len(regime_df_aligned),
            'avg_return': returns_df_aligned.loc[mask].mean().mean(),
            'avg_volatility': returns_df_aligned.loc[mask].std().mean(),
            'avg_prob': regime_df_aligned.loc[mask, regime_col].mean(),
            'n_periods': mask.sum()
        }
    return stats


def plot_regime_probabilities(regime_df: pd.DataFrame, returns_df: pd.DataFrame, save_dir: Path) -> None:
    """Plot regime probabilities over time (Stacked Area and Dominant Regime)."""
    ensure_directory(save_dir)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    regime_cols = [col for col in regime_df.columns if col.startswith('Regime_')]
    
    # Stacked area plot
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, col in enumerate(regime_cols):
        ax.fill_between(regime_df.index, 0, regime_df[col], alpha=0.5, label=f'Regime {i+1}', color=colors[i % len(colors)])
    ax.set_xlabel('Date'); ax.set_ylabel('Probability')
    ax.set_title('MoE Regime Probabilities Over Time')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_dir / 'regime_probabilities.png', dpi=150); plt.close()
    
    # Dominant regime scatter plot
    fig, ax = plt.subplots(figsize=(14, 4))
    dominant = regime_df['dominant_regime']
    colors_map = {i+1: colors[i % len(colors)] for i in range(len(regime_cols))}
    for regime in sorted(dominant.unique()):
        mask = dominant == regime
        ax.scatter(regime_df.index[mask], [regime] * mask.sum(), color=colors_map[regime], s=5, alpha=0.7, label=f'Regime {regime}')
    ax.set_xlabel('Date'); ax.set_ylabel('Dominant Regime')
    ax.set_title('Dominant Regime Over Time'); ax.set_yticks(list(colors_map.keys()))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_dir / 'dominant_regime.png', dpi=150); plt.close()
    logger.info(f"Regime figures saved to {save_dir}")


def print_regime_summary(stats: Dict) -> None:
    """Print regime summary to console."""
    print("\n" + "=" * 70); print("REGIME CHARACTERISTICS SUMMARY"); print("=" * 70)
    for regime, data in stats.items():
        print(f"\n{regime}:")
        print(f"  Frequency:        {data['frequency']:.2%}")
        print(f"  Number of periods: {data['n_periods']}")
        print(f"  Avg Return:       {data['avg_return']:.2f}%")
        print(f"  Avg Volatility:   {data['avg_volatility']:.2f}%")
        print(f"  Avg Probability:  {data['avg_prob']:.2%}")


def save_regime_summary(stats: Dict, save_path: Path) -> None:
    """Save regime summary to CSV."""
    df = pd.DataFrame(stats).T
    df.index.name = 'regime'
    df.to_csv(save_path)
    logger.info(f"Regime summary saved to {save_path}")


def analyze_moe_regimes(model, X: pd.DataFrame, returns_df: pd.DataFrame, dates: pd.DatetimeIndex, save_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """Complete regime analysis pipeline."""
    regime_df = extract_regime_data(model, X, dates)
    regime_stats = analyze_regime_characteristics(regime_df, returns_df)
    print_regime_summary(regime_stats)
    if save_dir:
        ensure_directory(save_dir)
        regime_df.to_csv(save_dir / 'regime_probabilities.csv')
        save_regime_summary(regime_stats, save_dir / 'regime_summary.csv')
        plot_regime_probabilities(regime_df, returns_df, save_dir / 'figures')
    return regime_df, regime_stats


# ============================================================================
# NEW PAPER-READY VISUALIZATION FUNCTIONS
# ============================================================================

def load_results_from_timestamp(timestamp: str) -> Dict:
    """
    Load all results from a specific run timestamp.
    
    Args:
        timestamp: String like '20260729_222829'
        
    Returns:
        Dictionary containing summary_df, predictions_dict, and config
    """
    results_dir = get_results_dir()
    
    # Load summary
    summary_path = results_dir / f'summary_{timestamp}.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    summary_df = pd.read_csv(summary_path, index_col=0)
    
    # Load config
    config_path = results_dir / f'config_{timestamp}.json'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Load predictions for key models
    pred_dir = results_dir / 'predictions' / timestamp
    predictions_dict = {}
    if pred_dir.exists():
        for model in ['moe', 'rolling_avg', 'linear', 'rf', 'momentum', 'persistence']:
            pred_file = pred_dir / f'{model}_predictions.csv'
            actual_file = pred_dir / f'{model}_actuals.csv'
            if pred_file.exists() and actual_file.exists():
                predictions_dict[model] = {
                    'predictions': pd.read_csv(pred_file, index_col=0, parse_dates=True),
                    'actuals': pd.read_csv(actual_file, index_col=0, parse_dates=True)
                }
    
    return {
        'summary': summary_df,
        'predictions': predictions_dict,
        'config': config,
        'timestamp': timestamp
    }


def plot_model_comparison(summary_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Generate bar charts comparing all models across key metrics.
    
    Args:
        summary_df: DataFrame from summary_{timestamp}.csv
        save_dir: Directory to save figures
    """
    ensure_directory(save_dir)
    
    metrics = ['sharpe', 'ann_return', 'max_drawdown', 'calmar', 'win_rate', 'rmse']
    titles = ['Sharpe Ratio', 'Annualized Return (%)', 'Max Drawdown (%)', 'Calmar Ratio', 'Win Rate', 'RMSE']
    ylabels = ['Sharpe', 'Return (%)', 'Drawdown (%)', 'Calmar', 'Win Rate', 'RMSE']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[i]
        data = summary_df[metric].sort_values(ascending=False)
        
        # For Max Drawdown and RMSE, lower is better, so we invert the visual logic
        if metric in ['max_drawdown', 'rmse']:
            data = data.sort_values(ascending=True)
            colors = ['#d62728' if val == data.min() else '#1f77b4' for val in data.values]
        else:
            colors = ['#2ca02c' if val == data.max() else '#1f77b4' for val in data.values]
        
        bars = ax.bar(data.index, data.values, color=colors)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison.png', dpi=150)
    plt.close()
    logger.info(f"Model comparison figure saved to {save_dir / 'model_comparison.png'}")


def plot_per_factor_rmse(predictions_dict: Dict, save_dir: Path) -> None:
    """
    Generate a grouped bar chart showing RMSE by factor for each model.
    
    Args:
        predictions_dict: Dictionary of model predictions/actuals
        save_dir: Directory to save figures
    """
    ensure_directory(save_dir)
    
    # Extract per-factor RMSE from the predictions files
    factor_rmse = {}
    for model_name, data in predictions_dict.items():
        preds = data['predictions'].values
        actuals = data['actuals'].values
        factors = data['predictions'].columns.tolist()
        
        rmse_list = []
        for i in range(preds.shape[1]):
            rmse = np.sqrt(np.mean((actuals[:, i] - preds[:, i]) ** 2))
            rmse_list.append(rmse)
        factor_rmse[model_name] = rmse_list
    
    # Create DataFrame
    df = pd.DataFrame(factor_rmse, index=factors)
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Per-Factor RMSE by Model', fontsize=14)
    ax.set_xlabel('Factor')
    ax.set_ylabel('RMSE')
    ax.legend(title='Model', bbox_to_anchor=(1, 1))
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'per_factor_rmse.png', dpi=150)
    plt.close()
    logger.info(f"Per-factor RMSE figure saved to {save_dir / 'per_factor_rmse.png'}")


def plot_cumulative_returns_comparison(predictions_dict: Dict, save_dir: Path) -> None:
    """
    Fixes the broken cumulative returns plot and compares MoE vs Rolling Avg vs Equal-Weight.
    
    Args:
        predictions_dict: Dictionary of model predictions/actuals
        save_dir: Directory to save figures
    """
    ensure_directory(save_dir)
    
    # Portfolio returns helper
    def get_portfolio_returns(model_name):
        if model_name not in predictions_dict:
            return None
        preds = predictions_dict[model_name]['predictions'].values
        actuals = predictions_dict[model_name]['actuals'].values
        
        # Magnitude-weighted long-only strategy (same as backtest)
        weights = np.where(preds > 0, preds, 0.0)
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums != 0)
        returns = np.sum(weights * actuals, axis=1)
        
        # Apply 10 bps cost (simplified)
        cost_decimal = 10 / 10000
        prev_weights = np.zeros_like(weights)
        prev_weights[0] = 1.0 / weights.shape[1]
        turnover = np.zeros(len(weights))
        for i in range(1, len(weights)):
            turnover[i] = np.sum(np.abs(weights[i] - weights[i-1])) / 2
        transaction_costs = turnover * cost_decimal
        returns = returns - transaction_costs
        
        return returns
    
    # Calculate cumulative returns
    dates = predictions_dict['moe']['predictions'].index
    
    # MoE
    moe_returns = get_portfolio_returns('moe')
    moe_cum = (1 + moe_returns).cumprod() if moe_returns is not None else pd.Series(1, index=dates)
    
    # Rolling Average
    rolling_returns = get_portfolio_returns('rolling_avg')
    rolling_cum = (1 + rolling_returns).cumprod() if rolling_returns is not None else pd.Series(1, index=dates)
    
    # Equal Weight (benchmark)
    equal_returns = predictions_dict['moe']['actuals'].mean(axis=1).values
    equal_cum = (1 + equal_returns).cumprod()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, moe_cum, label='MoE (Magnitude-Weighted)', linewidth=2, color='#2ca02c')
    ax.plot(dates, rolling_cum, label='Rolling Average', linewidth=2, color='#ff7f0e')
    ax.plot(dates, equal_cum, label='Equal-Weight Portfolio', linewidth=2, color='#1f77b4', linestyle='--')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Cumulative Returns Comparison (Out-of-Sample)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cumulative_returns.png', dpi=150)
    plt.close()
    logger.info(f"Fixed cumulative returns figure saved to {save_dir / 'cumulative_returns.png'}")


def plot_training_window_sensitivity(save_dir: Path) -> None:
    """
    Plot training window sensitivity using hardcoded data from TODO.md.
    
    Args:
        save_dir: Directory to save figures
    """
    ensure_directory(save_dir)
    
    # Data from your TODO.md (Day 4 - Afternoon Session)
    data = {
        'min_train': [60, 84, 96, 108, 120, 132],
        'sharpe': [0.7253, 1.2153, 1.4790, 1.3056, 1.7878, 1.7620],
        'return': [0.35, 41.45, 40.32, 39.24, 63.23, 90.42],
        'max_dd': [-44.41, -27.47, -13.74, -13.74, -13.74, -7.07]
    }
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sharpe
    axes[0].plot(df['min_train'], df['sharpe'], marker='o', linewidth=2, color='#1f77b4')
    axes[0].set_xlabel('Minimum Training Months')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title('Training Window Sensitivity: Sharpe')
    axes[0].grid(True, alpha=0.3)
    
    # Return
    axes[1].plot(df['min_train'], df['return'], marker='o', linewidth=2, color='#2ca02c')
    axes[1].set_xlabel('Minimum Training Months')
    axes[1].set_ylabel('Annualized Return (%)')
    axes[1].set_title('Training Window Sensitivity: Return')
    axes[1].grid(True, alpha=0.3)
    
    # Max Drawdown
    axes[2].plot(df['min_train'], df['max_dd'], marker='o', linewidth=2, color='#d62728')
    axes[2].set_xlabel('Minimum Training Months')
    axes[2].set_ylabel('Max Drawdown (%)')
    axes[2].set_title('Training Window Sensitivity: Max DD')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_window_sensitivity.png', dpi=150)
    plt.close()
    logger.info(f"Training window sensitivity figure saved to {save_dir / 'training_window_sensitivity.png'}")


def plot_vix_vs_fred_comparison(save_dir: Path) -> None:
    """
    Plot VIX vs FRED comparison using hardcoded data from TODO.md.
    
    Args:
        save_dir: Directory to save figures
    """
    ensure_directory(save_dir)
    
    # Data from your TODO.md (Day 4 - Afternoon Session, VIX vs FRED Comparative Analysis)
    data = {
        'Feature Set': ['VIX-only', 'FRED-enhanced'],
        'Sharpe': [1.7620, 0.4609],
        'Annual Return (%)': [90.42, 11.03],
        'Max Drawdown (%)': [-7.07, -37.64]
    }
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sharpe
    axes[0].bar(df['Feature Set'], df['Sharpe'], color=['#2ca02c', '#d62728'])
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title('VIX vs FRED: Sharpe Ratio')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Return
    axes[1].bar(df['Feature Set'], df['Annual Return (%)'], color=['#2ca02c', '#d62728'])
    axes[1].set_ylabel('Annual Return (%)')
    axes[1].set_title('VIX vs FRED: Return')
    axes[1].grid(True, axis='y', alpha=0.3)
    
    # Max Drawdown
    axes[2].bar(df['Feature Set'], df['Max Drawdown (%)'], color=['#2ca02c', '#d62728'])
    axes[2].set_ylabel('Max Drawdown (%)')
    axes[2].set_title('VIX vs FRED: Max Drawdown')
    axes[2].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'vix_vs_fred_comparison.png', dpi=150)
    plt.close()
    logger.info(f"VIX vs FRED comparison figure saved to {save_dir / 'vix_vs_fred_comparison.png'}")


def generate_paper_summary_table(summary_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Generate a single CSV containing all metrics needed for the paper.
    
    Args:
        summary_df: DataFrame from summary_{timestamp}.csv
        save_dir: Directory to save CSV
    """
    ensure_directory(save_dir)
    
    # Reorganize the summary DataFrame for the paper
    paper_df = summary_df.copy()
    paper_df = paper_df.round(4)
    
    # Save to CSV
    save_path = save_dir / 'paper_summary_table.csv'
    paper_df.to_csv(save_path)
    logger.info(f"Paper summary table saved to {save_path}")


def generate_all_paper_plots(timestamp: str, output_dir: Optional[Path] = None) -> None:
    """
    Master function to generate all paper-ready visualizations from a single timestamp.
    
    Args:
        timestamp: String like '20260729_222829'
        output_dir: Directory to save all figures (defaults to results/paper_figures/{timestamp})
    """
    logger.info(f"Generating paper plots for timestamp: {timestamp}")
    
    # Load data
    data = load_results_from_timestamp(timestamp)
    summary_df = data['summary']
    predictions_dict = data['predictions']
    
    # Set output directory
    if output_dir is None:
        output_dir = get_results_dir() / 'paper_figures' / timestamp
    ensure_directory(output_dir)
    
    logger.info(f"Saving all paper figures to: {output_dir}")
    
    # Generate all plots
    plot_model_comparison(summary_df, output_dir)
    plot_per_factor_rmse(predictions_dict, output_dir)
    plot_cumulative_returns_comparison(predictions_dict, output_dir)
    plot_training_window_sensitivity(output_dir)
    plot_vix_vs_fred_comparison(output_dir)
    generate_paper_summary_table(summary_df, output_dir)
    
    logger.info(f"All paper plots generated successfully in {output_dir}")


if __name__ == "__main__":
    # Example usage (update this timestamp to match your latest run)
    # generate_all_paper_plots('20260729_222829')
    print("Visualization module updated with paper-ready functions.")
    print("Run: generate_all_paper_plots('YOUR_TIMESTAMP') to generate all figures.")