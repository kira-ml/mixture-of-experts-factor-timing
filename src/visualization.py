"""
Visualization and analysis for MoE regime probabilities.
Generates plots and exports performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from src.utils import get_results_dir, ensure_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_regime_data(
    model,
    X: pd.DataFrame,
    dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Extract regime probabilities from fitted MoE model.
    
    Args:
        model: Fitted MoE model with predict_proba method
        X: Feature matrix used for predictions
        dates: Corresponding dates
        
    Returns:
        DataFrame with regime probabilities and dominant regime
    """
    # Clean X: replace inf with NaN, then fill with 0
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Get regime probabilities
    probs = model.predict_proba(X_clean)
    
    # Create DataFrame
    regime_df = pd.DataFrame(
        probs,
        index=dates,
        columns=[f'Regime_{i+1}' for i in range(probs.shape[1])]
    )
    
    # Add dominant regime (highest probability)
    regime_df['dominant_regime'] = np.argmax(probs, axis=1) + 1
    regime_df['dominant_prob'] = np.max(probs, axis=1)
    
    return regime_df


def analyze_regime_characteristics(
    regime_df: pd.DataFrame,
    returns_df: pd.DataFrame
) -> Dict:
    """
    Analyze characteristics of each regime.
    
    Args:
        regime_df: DataFrame with regime probabilities
        returns_df: DataFrame of factor returns
        
    Returns:
        Dictionary with regime statistics
    """
    n_regimes = regime_df.shape[1] - 2  # Exclude dominant_regime and dominant_prob
    stats = {}
    
    # Align indices between regime_df and returns_df
    common_idx = regime_df.index.intersection(returns_df.index)
    
    if len(common_idx) == 0:
        logger.warning("No overlapping indices between regime_df and returns_df")
        for i in range(n_regimes):
            regime_col = f'Regime_{i+1}'
            stats[regime_col] = {
                'frequency': 0,
                'avg_return': np.nan,
                'avg_volatility': np.nan,
                'avg_prob': 0,
                'n_periods': 0
            }
        return stats
    
    # Align both DataFrames
    regime_df_aligned = regime_df.loc[common_idx]
    returns_df_aligned = returns_df.loc[common_idx]
    
    for i in range(n_regimes):
        regime_col = f'Regime_{i+1}'
        # Get periods where this regime has highest probability
        mask = regime_df_aligned['dominant_regime'] == i + 1
        
        if mask.sum() == 0:
            stats[regime_col] = {
                'frequency': 0,
                'avg_return': np.nan,
                'avg_volatility': np.nan,
                'avg_prob': 0,
                'n_periods': 0
            }
            continue
        
        # Regime frequency
        freq = mask.sum() / len(regime_df_aligned)
        
        # Average returns during this regime
        regime_returns = returns_df_aligned.loc[mask].mean().mean()
        
        # Average volatility during this regime
        regime_vol = returns_df_aligned.loc[mask].std().mean()
        
        # Average probability of this regime
        avg_prob = regime_df_aligned.loc[mask, regime_col].mean()
        
        stats[regime_col] = {
            'frequency': freq,
            'avg_return': regime_returns,
            'avg_volatility': regime_vol,
            'avg_prob': avg_prob,
            'n_periods': mask.sum()
        }
    
    return stats


def plot_regime_probabilities(
    regime_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    save_dir: Path
) -> None:
    """
    Plot regime probabilities over time.
    
    Args:
        regime_df: DataFrame with regime probabilities
        returns_df: DataFrame of factor returns
        save_dir: Directory to save figures
    """
    ensure_directory(save_dir)
    
    # Figure 1: Regime probabilities over time
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    regime_cols = [col for col in regime_df.columns if col.startswith('Regime_')]
    
    for i, col in enumerate(regime_cols):
        ax.fill_between(
            regime_df.index,
            0,
            regime_df[col],
            alpha=0.5,
            label=f'Regime {i+1}',
            color=colors[i % len(colors)]
        )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Probability')
    ax.set_title('MoE Regime Probabilities Over Time')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'regime_probabilities.png', dpi=150)
    plt.close()
    
    # Figure 2: Dominant regime over time
    fig, ax = plt.subplots(figsize=(14, 4))
    
    dominant = regime_df['dominant_regime']
    colors_map = {i+1: colors[i % len(colors)] for i in range(len(regime_cols))}
    for regime in sorted(dominant.unique()):
        mask = dominant == regime
        ax.scatter(
            regime_df.index[mask],
            [regime] * mask.sum(),
            color=colors_map[regime],
            s=5,
            alpha=0.7,
            label=f'Regime {regime}'
        )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Dominant Regime')
    ax.set_title('Dominant Regime Over Time')
    ax.set_yticks(list(colors_map.keys()))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'dominant_regime.png', dpi=150)
    plt.close()
    
    # Figure 3: Cumulative returns comparison (MoE vs Equal Weight)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate cumulative returns
    cum_returns = (1 + returns_df.mean(axis=1)).cumprod()
    cum_returns.plot(ax=ax, label='Equal Weight Portfolio')
    
    # We need portfolio returns from MoE for comparison
    # This will be passed in separately
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Cumulative Returns Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cumulative_returns.png', dpi=150)
    plt.close()
    
    logger.info(f"Figures saved to {save_dir}")


def print_regime_summary(stats: Dict) -> None:
    """
    Print regime summary statistics to console.
    
    Args:
        stats: Dictionary from analyze_regime_characteristics
    """
    print("\n" + "=" * 70)
    print("REGIME CHARACTERISTICS SUMMARY")
    print("=" * 70)
    
    for regime, data in stats.items():
        print(f"\n{regime}:")
        print(f"  Frequency:        {data['frequency']:.2%}")
        print(f"  Number of periods: {data['n_periods']}")
        print(f"  Avg Return:       {data['avg_return']:.2f}%")
        print(f"  Avg Volatility:   {data['avg_volatility']:.2f}%")
        print(f"  Avg Probability:  {data['avg_prob']:.2%}")


def save_regime_summary(stats: Dict, save_path: Path) -> None:
    """
    Save regime summary statistics to CSV.
    
    Args:
        stats: Dictionary from analyze_regime_characteristics
        save_path: Path to save CSV
    """
    df = pd.DataFrame(stats).T
    df.index.name = 'regime'
    df.to_csv(save_path)
    logger.info(f"Regime summary saved to {save_path}")


def analyze_moe_regimes(
    model,
    X: pd.DataFrame,
    returns_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    save_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete regime analysis pipeline.
    
    Args:
        model: Fitted MoE model
        X: Feature matrix
        returns_df: DataFrame of factor returns
        dates: Dates corresponding to predictions
        save_dir: Directory to save outputs (optional)
        
    Returns:
        Tuple of (regime_df, regime_stats)
    """
    # Extract regime probabilities
    regime_df = extract_regime_data(model, X, dates)
    
    # Analyze regime characteristics
    regime_stats = analyze_regime_characteristics(regime_df, returns_df)
    
    # Print summary
    print_regime_summary(regime_stats)
    
    # Save if directory provided
    if save_dir:
        ensure_directory(save_dir)
        
        # Save regime probabilities
        regime_df.to_csv(save_dir / 'regime_probabilities.csv')
        logger.info(f"Regime probabilities saved to {save_dir / 'regime_probabilities.csv'}")
        
        # Save regime summary
        save_regime_summary(regime_stats, save_dir / 'regime_summary.csv')
        
        # Generate plots
        plot_regime_probabilities(regime_df, returns_df, save_dir / 'figures')
    
    return regime_df, regime_stats


if __name__ == "__main__":
    # Example usage
    print("Regime analysis module loaded.")
    print("Import and use analyze_moe_regimes() with a fitted MoE model.")