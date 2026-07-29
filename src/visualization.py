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
import matplotlib.ticker as mtick

from src.utils import get_results_dir, ensure_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL ACADEMIC PLOTTING STYLE
# ============================================================================
def _set_academic_style():
    """Set global Matplotlib parameters for a high-quality academic paper look."""
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linewidth': 0.8,
    })

_set_academic_style()

# High-contrast, colorblind-friendly palette (Tableau 10)
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
BEST_COLOR = '#2ca02c'  # Green for best
WORST_COLOR = '#d62728' # Red for worst
NEUTRAL_COLOR = '#1f77b4' # Blue for middle

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
    regime_cols = [col for col in regime_df.columns if col.startswith('Regime_')]
    colors = COLORS[:len(regime_cols)]
    
    # Stacked area plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(regime_df.index, [regime_df[col] for col in regime_cols], 
                 labels=[f'Regime {i+1}' for i in range(len(regime_cols))], 
                 colors=colors, alpha=0.85)
    ax.set_xlabel('Date'); ax.set_ylabel('Probability')
    ax.set_title('MoE Regime Probabilities Over Time', fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(save_dir / 'regime_probabilities.png'); plt.close()
    
    # Dominant regime scatter plot
    fig, ax = plt.subplots(figsize=(14, 4))
    dominant = regime_df['dominant_regime']
    colors_map = {i+1: colors[i % len(colors)] for i in range(len(regime_cols))}
    
    # Add small jitter for Y-axis readability
    for regime in sorted(dominant.unique()):
        mask = dominant == regime
        y_vals = np.ones(mask.sum()) * regime + np.random.normal(0, 0.03, mask.sum())
        ax.scatter(regime_df.index[mask], y_vals, color=colors_map[regime], s=10, alpha=0.8, label=f'Regime {regime}', edgecolors='white')
    
    ax.set_xlabel('Date'); ax.set_ylabel('Dominant Regime')
    ax.set_title('Dominant Regime Over Time', fontweight='bold')
    ax.set_yticks(list(colors_map.keys())); ax.set_ylim(0.5, len(colors_map)+0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.tight_layout(); plt.savefig(save_dir / 'dominant_regime.png'); plt.close()
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
    results_dir = get_results_dir()
    summary_path = results_dir / f'summary_{timestamp}.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    summary_df = pd.read_csv(summary_path, index_col=0)
    
    config_path = results_dir / f'config_{timestamp}.json'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
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
    return {'summary': summary_df, 'predictions': predictions_dict, 'config': config, 'timestamp': timestamp}


def plot_model_comparison(summary_df: pd.DataFrame, save_dir: Path) -> None:
    """Generate academic bar charts comparing all models."""
    ensure_directory(save_dir)
    
    metrics = ['sharpe', 'ann_return', 'max_drawdown', 'calmar', 'win_rate', 'rmse']
    titles = ['Sharpe Ratio', 'Annualized Return (%)', 'Max Drawdown (%)', 'Calmar Ratio', 'Win Rate', 'RMSE']
    ylabels = ['Sharpe', 'Return (%)', 'Drawdown (%)', 'Calmar', 'Win Rate', 'RMSE']
    ascending_sort = [False, False, True, False, False, True]  # True = lower is better (Drawdown, RMSE)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (metric, title, ylabel, asc) in enumerate(zip(metrics, titles, ylabels, ascending_sort)):
        ax = axes[i]
        
        # Sort data logically
        if asc:
            data = summary_df[metric].sort_values(ascending=True)
        else:
            data = summary_df[metric].sort_values(ascending=False)
            
        # Color scheme: Best=Green, Worst=Red, Rest=Blue
        colors = []
        for idx, val in enumerate(data.values):
            if idx == 0: colors.append(BEST_COLOR)
            elif idx == len(data) - 1: colors.append(WORST_COLOR)
            else: colors.append(NEUTRAL_COLOR)
        
        bars = ax.bar(data.index, data.values, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels with smaller font
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height > 0 else 'top'
            offset = 0.5 if height > 0 else -0.5
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   f'{height:.2f}', ha='center', va=va, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison.png')
    plt.close()
    logger.info(f"Model comparison figure saved to {save_dir / 'model_comparison.png'}")


def plot_per_factor_rmse(predictions_dict: Dict, save_dir: Path) -> None:
    """
    Generate a HEATMAP showing RMSE by factor for each model.
    Heatmaps are far more aesthetic and compact for academic ML papers than 36-bar charts.
    """
    ensure_directory(save_dir)
    
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
    
    df = pd.DataFrame(factor_rmse, index=factors)
    
    # Sort rows/cols for better readability
    df = df.reindex(df.mean(axis=1).sort_values().index, axis=0)
    df = df.reindex(df.mean(axis=0).sort_values().index, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, 
                linewidths=0.5, cbar_kws={'label': 'RMSE'})
    ax.set_title('Per-Factor RMSE by Model (Heatmap)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Model'); ax.set_ylabel('Factor')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / 'per_factor_rmse.png')
    plt.close()
    logger.info(f"Per-factor RMSE heatmap saved to {save_dir / 'per_factor_rmse.png'}")


def plot_cumulative_returns_comparison(predictions_dict: Dict, save_dir: Path) -> None:
    """Fixed cumulative returns plot showing full 2019-2026 timeline."""
    ensure_directory(save_dir)
    
    def get_portfolio_returns(model_name):
        if model_name not in predictions_dict: return None
        preds = predictions_dict[model_name]['predictions'].values
        actuals = predictions_dict[model_name]['actuals'].values
        
        weights = np.where(preds > 0, preds, 0.0)
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums != 0)
        returns = np.sum(weights * actuals, axis=1)
        
        # Cost logic
        cost_decimal = 10 / 10000
        prev_weights = np.zeros_like(weights)
        prev_weights[0] = 1.0 / weights.shape[1]
        turnover = np.zeros(len(weights))
        for i in range(1, len(weights)):
            turnover[i] = np.sum(np.abs(weights[i] - weights[i-1])) / 2
        transaction_costs = turnover * cost_decimal
        return returns - transaction_costs
    
    dates = predictions_dict['moe']['predictions'].index
    
    # Get returns
    moe_returns = get_portfolio_returns('moe')
    rolling_returns = get_portfolio_returns('rolling_avg')
    equal_returns = predictions_dict['moe']['actuals'].mean(axis=1).values
    
    # Calculate cumulative products (start at 1.0)
    moe_cum = (1 + moe_returns).cumprod()
    rolling_cum = (1 + rolling_returns).cumprod()
    equal_cum = (1 + equal_returns).cumprod()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, moe_cum, label='MoE (Magnitude-Weighted)', linewidth=2.5, color=BEST_COLOR)
    ax.plot(dates, rolling_cum, label='Rolling Average (Baseline)', linewidth=2, color=NEUTRAL_COLOR, linestyle='--')
    ax.plot(dates, equal_cum, label='Equal-Weight Portfolio (Benchmark)', linewidth=2, color=WORST_COLOR, linestyle=':')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (Starting at 1.0)')
    ax.set_title('Cumulative Returns Comparison (Out-of-Sample)', fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, moe_cum.max() * 1.1)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cumulative_returns.png')
    plt.close()
    logger.info(f"Cumulative returns figure saved to {save_dir / 'cumulative_returns.png'}")


def plot_training_window_sensitivity(save_dir: Path) -> None:
    """Plot training window sensitivity with academic styling."""
    ensure_directory(save_dir)
    
    data = {
        'min_train': [60, 84, 96, 108, 120, 132],
        'sharpe': [0.7253, 1.2153, 1.4790, 1.3056, 1.7878, 1.7620],
        'return': [0.35, 41.45, 40.32, 39.24, 63.23, 90.42],
        'max_dd': [-44.41, -27.47, -13.74, -13.74, -13.74, -7.07]
    }
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sharpe
    axes[0].plot(df['min_train'], df['sharpe'], marker='o', markersize=6, linewidth=2, color=NEUTRAL_COLOR)
    axes[0].fill_between(df['min_train'], df['sharpe'], alpha=0.15, color=NEUTRAL_COLOR)
    axes[0].set_xlabel('Minimum Training Months'); axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title('Training Window Sensitivity: Sharpe', fontweight='bold')
    
    # Return
    axes[1].plot(df['min_train'], df['return'], marker='o', markersize=6, linewidth=2, color=BEST_COLOR)
    axes[1].fill_between(df['min_train'], df['return'], alpha=0.15, color=BEST_COLOR)
    axes[1].set_xlabel('Minimum Training Months'); axes[1].set_ylabel('Annualized Return (%)')
    axes[1].set_title('Training Window Sensitivity: Return', fontweight='bold')
    
    # Max Drawdown (inverted so -7% is visually higher than -44%)
    axes[2].plot(df['min_train'], df['max_dd'] * -1, marker='o', markersize=6, linewidth=2, color=WORST_COLOR)
    axes[2].fill_between(df['min_train'], df['max_dd'] * -1, alpha=0.15, color=WORST_COLOR)
    axes[2].set_xlabel('Minimum Training Months'); axes[2].set_ylabel('Max Drawdown (Absolute %)')
    axes[2].set_title('Training Window Sensitivity: Max Drawdown', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_window_sensitivity.png')
    plt.close()
    logger.info(f"Training window sensitivity figure saved to {save_dir / 'training_window_sensitivity.png'}")


def plot_vix_vs_fred_comparison(save_dir: Path) -> None:
    """Plot VIX vs FRED comparison with academic styling."""
    ensure_directory(save_dir)
    
    data = {
        'Feature Set': ['VIX-only', 'FRED-enhanced'],
        'Sharpe': [1.7620, 0.4609],
        'Annual Return (%)': [90.42, 11.03],
        'Max Drawdown (%)': [-7.07, -37.64]
    }
    df = pd.DataFrame(data)
    colors = [BEST_COLOR, WORST_COLOR]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sharpe
    axes[0].bar(df['Feature Set'], df['Sharpe'], color=colors, edgecolor='white', linewidth=1.5, zorder=3)
    axes[0].set_ylabel('Sharpe Ratio'); axes[0].set_title('VIX vs FRED: Sharpe Ratio', fontweight='bold')
    
    # Return
    axes[1].bar(df['Feature Set'], df['Annual Return (%)'], color=colors, edgecolor='white', linewidth=1.5, zorder=3)
    axes[1].set_ylabel('Annual Return (%)'); axes[1].set_title('VIX vs FRED: Return', fontweight='bold')
    
    # Max Drawdown
    axes[2].bar(df['Feature Set'], df['Max Drawdown (%)'], color=colors, edgecolor='white', linewidth=1.5, zorder=3)
    axes[2].set_ylabel('Max Drawdown (%)'); axes[2].set_title('VIX vs FRED: Max Drawdown', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'vix_vs_fred_comparison.png')
    plt.close()
    logger.info(f"VIX vs FRED comparison figure saved to {save_dir / 'vix_vs_fred_comparison.png'}")


def generate_paper_summary_table(summary_df: pd.DataFrame, save_dir: Path) -> None:
    ensure_directory(save_dir)
    paper_df = summary_df.copy().round(4)
    save_path = save_dir / 'paper_summary_table.csv'
    paper_df.to_csv(save_path)
    logger.info(f"Paper summary table saved to {save_path}")


def generate_all_paper_plots(timestamp: str, output_dir: Optional[Path] = None) -> None:
    logger.info(f"Generating paper plots for timestamp: {timestamp}")
    data = load_results_from_timestamp(timestamp)
    summary_df = data['summary']
    predictions_dict = data['predictions']
    
    if output_dir is None:
        output_dir = get_results_dir() / 'paper_figures' / timestamp
    ensure_directory(output_dir)
    
    logger.info(f"Saving all paper figures to: {output_dir}")
    plot_model_comparison(summary_df, output_dir)
    plot_per_factor_rmse(predictions_dict, output_dir)
    plot_cumulative_returns_comparison(predictions_dict, output_dir)
    plot_training_window_sensitivity(output_dir)
    plot_vix_vs_fred_comparison(output_dir)
    generate_paper_summary_table(summary_df, output_dir)
    logger.info(f"All paper plots generated successfully in {output_dir}")


if __name__ == "__main__":
    # generate_all_paper_plots('20260729_222829')
    print("Visualization module updated with high-end academic paper-ready functions.")
    print("Run: generate_all_paper_plots('YOUR_TIMESTAMP') to generate all figures.")