"""
Visualization for MoE Regime-Switching Factor Timing Paper.
Generates publication-quality figures for the mini research paper.

Usage:
    python src/visualization.py --timestamp 20260730_013120
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
import argparse

from src.utils import get_results_dir, ensure_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: STYLE CONFIGURATION
# ============================================================================

def set_academic_style():
    """Set publication-quality Matplotlib style."""
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
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
    })


set_academic_style()

# Color palette (colorblind-friendly)
COLORS = {
    'moe': '#2ca02c',          # Green
    'momentum': '#ff7f0e',     # Orange
    'rolling_avg': '#1f77b4',  # Blue
    'linear': '#9467bd',       # Purple
    'rf': '#8c564b',           # Brown
    'persistence': '#d62728',  # Red
}

REGIME_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

BEST_COLOR = '#2ca02c'
WORST_COLOR = '#d62728'
NEUTRAL_COLOR = '#1f77b4'

MODEL_ORDER = ['moe', 'momentum', 'rolling_avg', 'linear', 'rf', 'persistence']


# ============================================================================
# PART 2: DATA LOADING
# ============================================================================

def load_data(timestamp: str) -> Dict:
    """
    Load all data for a given timestamp.
    
    Args:
        timestamp: Run timestamp (e.g., '20260730_013120')
        
    Returns:
        Dict with keys: summary, config, predictions, portfolio_returns
    """
    results_dir = get_results_dir()
    
    # Load summary
    summary_path = results_dir / f'summary_{timestamp}.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    summary_df = pd.read_csv(summary_path, index_col=0)
    
    # Load config
    config_path = results_dir / f'config_{timestamp}.json'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Load predictions and portfolio returns
    pred_dir = results_dir / 'predictions' / timestamp
    predictions_dict = {}
    portfolio_returns_dict = {}
    
    if pred_dir.exists():
        for model in MODEL_ORDER:
            pred_file = pred_dir / f'{model}_predictions.csv'
            actual_file = pred_dir / f'{model}_actuals.csv'
            port_file = pred_dir / f'{model}_portfolio_returns.csv'
            
            if pred_file.exists() and actual_file.exists():
                preds = pd.read_csv(pred_file, index_col=0, parse_dates=True)
                actuals = pd.read_csv(actual_file, index_col=0, parse_dates=True)
                predictions_dict[model] = {'predictions': preds, 'actuals': actuals}
            
            if port_file.exists():
                port_df = pd.read_csv(port_file, index_col=0, parse_dates=True)
                portfolio_returns_dict[model] = port_df['portfolio_returns']
    
    return {
        'summary': summary_df,
        'config': config,
        'predictions': predictions_dict,
        'portfolio_returns': portfolio_returns_dict,
        'timestamp': timestamp
    }


# ============================================================================
# PART 3: REGIME ANALYSIS (From fitted MoE model)
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


def analyze_moe_regimes(model, X: pd.DataFrame, returns_df: pd.DataFrame, dates: pd.DatetimeIndex, save_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """Complete regime analysis pipeline."""
    regime_df = extract_regime_data(model, X, dates)
    regime_stats = analyze_regime_characteristics(regime_df, returns_df)
    
    if save_dir:
        ensure_directory(save_dir)
        regime_df.to_csv(save_dir / 'regime_probabilities.csv')
        df = pd.DataFrame(regime_stats).T
        df.index.name = 'regime'
        df.to_csv(save_dir / 'regime_summary.csv')
    
    return regime_df, regime_stats


# ============================================================================
# PART 4: PAPER FIGURES
# ============================================================================

def plot_model_comparison(data: Dict, save_dir: Path) -> None:
    """
    Figure 1: Model comparison bar chart.
    Shows Sharpe, Return, Drawdown, Calmar, Win Rate, RMSE.
    """
    ensure_directory(save_dir)
    summary_df = data['summary']
    
    metrics = ['sharpe', 'ann_return', 'max_drawdown', 'calmar', 'win_rate', 'rmse']
    titles = ['Sharpe Ratio', 'Annualized Return (%)', 'Max Drawdown (%)', 'Calmar Ratio', 'Win Rate', 'RMSE']
    ylabels = ['Sharpe', 'Return (%)', 'Drawdown (%)', 'Calmar', 'Win Rate', 'RMSE']
    ascending = [False, False, True, False, False, True]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (metric, title, ylabel, asc) in enumerate(zip(metrics, titles, ylabels, ascending)):
        ax = axes[i]
        data_series = summary_df[metric].sort_values(ascending=asc)
        
        # Color: Best=Green, Worst=Red, Rest=Blue
        colors = []
        for idx, val in enumerate(data_series.values):
            if idx == 0:
                colors.append(BEST_COLOR)
            elif idx == len(data_series) - 1:
                colors.append(WORST_COLOR)
            else:
                colors.append(NEUTRAL_COLOR)
        
        bars = ax.bar(data_series.index, data_series.values, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        
        # Value labels
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height > 0 else 'top'
            offset = 0.5 if height > 0 else -0.5
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   f'{height:.2f}', ha='center', va=va, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison.png')
    plt.close()
    logger.info(f"Figure 1 saved: {save_dir / 'model_comparison.png'}")


def plot_cumulative_returns(data: Dict, save_dir: Path) -> None:
    """
    Figure 2: Cumulative returns comparison.
    Shows MoE vs Rolling Average vs Equal-Weight.
    CORRECTED: Plots only the dates that exist in the out-of-sample data.
    """
    ensure_directory(save_dir)
    
    portfolio_returns = data['portfolio_returns']
    
    # Get returns for each model
    moe_returns = portfolio_returns.get('moe')
    rolling_returns = portfolio_returns.get('rolling_avg')
    
    if moe_returns is None or rolling_returns is None:
        logger.warning("Portfolio returns not found. Skipping cumulative returns plot.")
        return
    
    # Convert percentage to decimal if needed
    def to_decimal(series):
        if series is None:
            return None
        if np.nanmax(np.abs(series.values)) > 1.5:
            return series / 100.0
        return series
    
    moe_returns = to_decimal(moe_returns)
    rolling_returns = to_decimal(rolling_returns)
    
    # Equal-weight: average of all factors from actuals
    actuals = data['predictions'].get('moe', {}).get('actuals')
    if actuals is not None:
        equal_returns = np.nanmean(actuals.values, axis=1)
        equal_returns = to_decimal(pd.Series(equal_returns, index=actuals.index))
    else:
        equal_returns = None
    
    # Find the common index across all available series
    common_index = moe_returns.index
    if rolling_returns is not None:
        common_index = common_index.intersection(rolling_returns.index)
    if equal_returns is not None:
        common_index = common_index.intersection(equal_returns.index)
    
    # Filter to common dates and compute cumulative products
    moe_cum = (1 + moe_returns.loc[common_index]).cumprod()
    rolling_cum = (1 + rolling_returns.loc[common_index]).cumprod()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(moe_cum.index, moe_cum, label='MoE (Magnitude-Weighted)', linewidth=2.5, color=BEST_COLOR)
    ax.plot(rolling_cum.index, rolling_cum, label='Rolling Average (Baseline)', linewidth=2, color=NEUTRAL_COLOR, linestyle='--')
    
    if equal_returns is not None:
        equal_cum = (1 + equal_returns.loc[common_index]).cumprod()
        ax.plot(equal_cum.index, equal_cum, label='Equal-Weight (Benchmark)', linewidth=2, color=WORST_COLOR, linestyle=':')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (Starting at 1.0)')
    ax.set_title('Cumulative Returns Comparison (Out-of-Sample)', fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    
    # Dynamic y-axis
    all_vals = [moe_cum, rolling_cum]
    if equal_returns is not None:
        all_vals.append(equal_cum)
    max_val = max([v.max() for v in all_vals])
    ax.set_ylim(0, max_val * 1.1)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cumulative_returns.png')
    plt.close()
    logger.info(f"Figure 2 saved: {save_dir / 'cumulative_returns.png'}")


def plot_per_factor_rmse(data: Dict, save_dir: Path) -> None:
    """
    Figure 3: Per-factor RMSE heatmap.
    Shows which factors are hard/easy to predict for each model.
    """
    ensure_directory(save_dir)
    
    predictions = data['predictions']
    factor_rmse = {}
    
    for model_name, model_data in predictions.items():
        preds = model_data['predictions'].values
        actuals = model_data['actuals'].values
        factors = model_data['predictions'].columns.tolist()
        
        rmse_list = []
        for i in range(preds.shape[1]):
            rmse = np.sqrt(np.mean((actuals[:, i] - preds[:, i]) ** 2))
            rmse_list.append(rmse)
        factor_rmse[model_name] = rmse_list
    
    df = pd.DataFrame(factor_rmse, index=factors)
    df = df.reindex(df.mean(axis=1).sort_values().index, axis=0)
    df = df.reindex(df.mean(axis=0).sort_values().index, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, 
                linewidths=0.5, cbar_kws={'label': 'RMSE'})
    ax.set_title('Per-Factor RMSE by Model', fontweight='bold', fontsize=12)
    ax.set_xlabel('Model')
    ax.set_ylabel('Factor')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / 'per_factor_rmse.png')
    plt.close()
    logger.info(f"Figure 3 saved: {save_dir / 'per_factor_rmse.png'}")


def plot_rmse_vs_sharpe(data: Dict, save_dir: Path) -> None:
    """
    Figure 4: RMSE vs Sharpe scatter plot.
    Shows trade-off between predictive accuracy and investment performance.
    """
    ensure_directory(save_dir)
    
    summary_df = data['summary']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter
    for model in summary_df.index:
        rmse = summary_df.loc[model, 'rmse']
        sharpe = summary_df.loc[model, 'sharpe']
        color = BEST_COLOR if model == 'moe' else NEUTRAL_COLOR
        ax.scatter(rmse, sharpe, s=120, color=color, zorder=3, edgecolors='white', linewidth=1.5)
        ax.annotate(model, (rmse, sharpe), xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('RMSE (Predictive Accuracy)', fontsize=12)
    ax.set_ylabel('Sharpe Ratio (Investment Performance)', fontsize=12)
    ax.set_title('RMSE vs Sharpe: Predictive Accuracy vs Investment Performance', fontweight='bold', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'rmse_vs_sharpe.png')
    plt.close()
    logger.info(f"Figure 4 saved: {save_dir / 'rmse_vs_sharpe.png'}")


def plot_regime_probabilities(regime_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Figure 5: Regime probabilities stacked area chart.
    """
    ensure_directory(save_dir)
    regime_cols = [col for col in regime_df.columns if col.startswith('Regime_')]
    colors = REGIME_COLORS[:len(regime_cols)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(regime_df.index, [regime_df[col] for col in regime_cols], 
                 labels=[f'Regime {i+1}' for i in range(len(regime_cols))], 
                 colors=colors, alpha=0.85)
    ax.set_xlabel('Date')
    ax.set_ylabel('Probability')
    ax.set_title('MoE Regime Probabilities Over Time', fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_dir / 'regime_probabilities.png')
    plt.close()
    logger.info(f"Figure 5 saved: {save_dir / 'regime_probabilities.png'}")


def plot_dominant_regime(regime_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Figure 6: Dominant regime over time.
    """
    ensure_directory(save_dir)
    regime_cols = [col for col in regime_df.columns if col.startswith('Regime_')]
    colors = REGIME_COLORS[:len(regime_cols)]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    dominant = regime_df['dominant_regime']
    colors_map = {i+1: colors[i % len(colors)] for i in range(len(regime_cols))}
    
    for regime in sorted(dominant.unique()):
        mask = dominant == regime
        y_vals = np.ones(mask.sum()) * regime + np.random.normal(0, 0.03, mask.sum())
        ax.scatter(regime_df.index[mask], y_vals, color=colors_map[regime], 
                   s=10, alpha=0.8, label=f'Regime {regime}', edgecolors='white')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Dominant Regime')
    ax.set_title('Dominant Regime Over Time', fontweight='bold')
    ax.set_yticks(list(colors_map.keys()))
    ax.set_ylim(0.5, len(colors_map) + 0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.tight_layout()
    plt.savefig(save_dir / 'dominant_regime.png')
    plt.close()
    logger.info(f"Figure 6 saved: {save_dir / 'dominant_regime.png'}")


def plot_regime_characteristics(regime_stats: Dict, save_dir: Path) -> None:
    """
    Figure 7: Regime characteristics scatter plot (Return vs Volatility).
    """
    ensure_directory(save_dir)
    
    df = pd.DataFrame(regime_stats).T
    df = df.dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, row in df.iterrows():
        size = row['frequency'] * 1000 + 100
        ax.scatter(row['avg_volatility'], row['avg_return'], 
                   s=size, color=REGIME_COLORS[int(i.split('_')[1]) - 1], 
                   alpha=0.7, edgecolors='black', linewidth=1, zorder=3)
        ax.annotate(i, (row['avg_volatility'], row['avg_return']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Average Volatility (%)', fontsize=12)
    ax.set_ylabel('Average Return (%)', fontsize=12)
    ax.set_title('Regime Characteristics: Return vs Volatility', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'regime_characteristics.png')
    plt.close()
    logger.info(f"Figure 7 saved: {save_dir / 'regime_characteristics.png'}")


def generate_paper_summary_table(data: Dict, save_dir: Path) -> None:
    """Save summary table for the paper."""
    ensure_directory(save_dir)
    summary_df = data['summary']
    paper_df = summary_df.round(4)
    save_path = save_dir / 'paper_summary_table.csv'
    paper_df.to_csv(save_path)
    logger.info(f"Summary table saved: {save_path}")


# ============================================================================
# PART 5: ORCHESTRATOR
# ============================================================================

def generate_all_figures(timestamp: str, output_dir: Optional[Path] = None) -> None:
    """
    Generate all paper figures for a given timestamp.
    
    Args:
        timestamp: Run timestamp (e.g., '20260730_013120')
        output_dir: Optional output directory
    """
    logger.info(f"Generating paper figures for timestamp: {timestamp}")
    
    # Load data
    data = load_data(timestamp)
    
    # Set output directory
    if output_dir is None:
        output_dir = get_results_dir() / 'paper_figures' / timestamp
    ensure_directory(output_dir)
    
    logger.info(f"Saving figures to: {output_dir}")
    
    # Generate figures
    plot_model_comparison(data, output_dir)
    plot_cumulative_returns(data, output_dir)
    plot_per_factor_rmse(data, output_dir)
    plot_rmse_vs_sharpe(data, output_dir)
    generate_paper_summary_table(data, output_dir)
    
    # Regime figures (if regime data exists)
    regime_dir = get_results_dir() / 'regime_analysis'
    if regime_dir.exists():
        regime_df_path = regime_dir / 'regime_probabilities.csv'
        regime_summary_path = regime_dir / 'regime_summary.csv'
        if regime_df_path.exists():
            regime_df = pd.read_csv(regime_df_path, index_col=0, parse_dates=True)
            plot_regime_probabilities(regime_df, output_dir)
            plot_dominant_regime(regime_df, output_dir)
        if regime_summary_path.exists():
            regime_stats = pd.read_csv(regime_summary_path, index_col=0).to_dict(orient='index')
            plot_regime_characteristics(regime_stats, output_dir)
    
    logger.info(f"All figures generated successfully in {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures for MoE Factor Timing project.")
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Run timestamp (e.g., 20260730_013120)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Optional output directory path")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir) if args.output_dir else None
    generate_all_figures(args.timestamp, output_path)