"""
Evaluation metrics for model comparison.
Week 1 MVP: Simple predictive and investment performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        float: RMSE value
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        return np.nan
    
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        float: MAE value
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        return np.nan
    
    return np.mean(np.abs(y_true - y_pred))


def annualized_return(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """
    Calculate annualized return from monthly returns.
    
    Args:
        returns: Array of returns (in percentage or decimal)
        periods_per_year: Number of periods in a year (default: 12 for monthly)
        
    Returns:
        float: Annualized return (same unit as input)
    """
    returns = np.array(returns).flatten()
    
    if len(returns) == 0:
        return np.nan
    
    # Convert to decimal if returns are in percentage
    if np.abs(returns).max() > 1:
        returns = returns / 100
    
    # Calculate compound annual return
    total_return = np.prod(1 + returns) - 1
    n_years = len(returns) / periods_per_year
    
    if n_years <= 0:
        return np.nan
    
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Convert back to percentage if input was percentage
    if np.abs(returns).max() < 1:
        return ann_return * 100
    else:
        return ann_return


def annualized_volatility(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """
    Calculate annualized volatility from monthly returns.
    
    Args:
        returns: Array of returns (in percentage or decimal)
        periods_per_year: Number of periods in a year (default: 12 for monthly)
        
    Returns:
        float: Annualized volatility (same unit as input)
    """
    returns = np.array(returns).flatten()
    
    if len(returns) < 2:
        return np.nan
    
    # Check if returns are in percentage
    is_percentage = np.abs(returns).max() > 1
    
    # Convert to decimal for calculation
    if is_percentage:
        returns = returns / 100
    
    # Calculate standard deviation
    vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    
    # Convert back to percentage if input was percentage
    if is_percentage:
        return vol * 100
    else:
        return vol


def sharpe_ratio(
    returns: np.ndarray, 
    risk_free_rate: float = 0.0, 
    periods_per_year: int = 12
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of returns (in percentage or decimal)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year (default: 12 for monthly)
        
    Returns:
        float: Sharpe ratio
    """
    returns = np.array(returns).flatten()
    
    if len(returns) < 2:
        return np.nan
    
    # Check if returns are in percentage
    is_percentage = np.abs(returns).max() > 1
    
    # Convert to decimal for calculation
    if is_percentage:
        returns = returns / 100
        risk_free_rate = risk_free_rate / 100
    
    # Convert risk-free rate to monthly
    risk_free_monthly = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - risk_free_monthly
    
    # Calculate Sharpe ratio
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return np.nan
    
    sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
    
    return sharpe


def maximum_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Array of returns (in percentage or decimal)
        
    Returns:
        float: Maximum drawdown (in same units as returns, negative value)
    """
    returns = np.array(returns).flatten()
    
    if len(returns) == 0:
        return np.nan
    
    # Check if returns are in percentage
    is_percentage = np.abs(returns).max() > 1
    
    # Convert to decimal for calculation
    if is_percentage:
        returns = returns / 100
    
    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdowns
    drawdowns = (cumulative - running_max) / running_max
    
    # Maximum drawdown
    max_dd = np.min(drawdowns)
    
    # Convert back to percentage if input was percentage
    if is_percentage:
        return max_dd * 100
    else:
        return max_dd


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """
    Calculate Calmar ratio (annualized return / maximum drawdown).
    
    Args:
        returns: Array of returns (in percentage or decimal)
        periods_per_year: Number of periods in a year (default: 12 for monthly)
        
    Returns:
        float: Calmar ratio
    """
    returns = np.array(returns).flatten()
    
    if len(returns) < 2:
        return np.nan
    
    ann_return = annualized_return(returns, periods_per_year)
    max_dd = maximum_drawdown(returns)
    
    if max_dd == 0:
        return np.nan
    
    # Calmar = annualized return / absolute maximum drawdown
    # Both should be in same units (percentage or decimal)
    return ann_return / abs(max_dd)


def win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of positive returns).
    
    Args:
        returns: Array of returns
        
    Returns:
        float: Win rate (0 to 1)
    """
    returns = np.array(returns).flatten()
    
    if len(returns) == 0:
        return np.nan
    
    positive = np.sum(returns > 0)
    return positive / len(returns)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    factor_names: Optional[List[str]] = None,
    returns: Optional[np.ndarray] = None,
    periods_per_year: int = 12
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate predictions with comprehensive metrics.
    
    Args:
        y_true: Actual values (predictions target)
        y_pred: Predicted values
        factor_names: Names of factors (for per-factor metrics)
        returns: Actual returns for investment metrics (if different from y_true)
        periods_per_year: Number of periods in a year (default: 12)
        
    Returns:
        Dictionary with metrics:
            - rmse: Overall RMSE
            - mae: Overall MAE
            - by_factor: Per-factor metrics (if factor_names provided)
            - investment: Investment metrics (if returns provided)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle single or multi-dimensional
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_samples, n_factors = y_true.shape
    
    if n_samples == 0:
        logger.warning("No samples to evaluate")
        return {}
    
    results = {}
    
    # Overall predictive metrics
    results['rmse'] = rmse(y_true.flatten(), y_pred.flatten())
    results['mae'] = mae(y_true.flatten(), y_pred.flatten())
    
    # Per-factor metrics
    if factor_names and len(factor_names) == n_factors:
        by_factor = {}
        for i, name in enumerate(factor_names):
            by_factor[name] = {
                'rmse': rmse(y_true[:, i], y_pred[:, i]),
                'mae': mae(y_true[:, i], y_pred[:, i]),
            }
        results['by_factor'] = by_factor
    else:
        # Use default names
        by_factor = {}
        for i in range(n_factors):
            by_factor[f'Factor_{i+1}'] = {
                'rmse': rmse(y_true[:, i], y_pred[:, i]),
                'mae': mae(y_true[:, i], y_pred[:, i]),
            }
        results['by_factor'] = by_factor
    
    # Investment metrics (if returns provided)
    if returns is not None:
        returns = np.array(returns).flatten()
        
        if len(returns) > 0:
            investment_metrics = {
                'annualized_return': annualized_return(returns, periods_per_year),
                'annualized_volatility': annualized_volatility(returns, periods_per_year),
                'sharpe_ratio': sharpe_ratio(returns, periods_per_year=periods_per_year),
                'maximum_drawdown': maximum_drawdown(returns),
                'calmar_ratio': calmar_ratio(returns, periods_per_year),
                'win_rate': win_rate(returns),
            }
            results['investment'] = investment_metrics
    
    return results


def compare_models(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    factor_names: Optional[List[str]] = None,
    returns: Optional[np.ndarray] = None,
    periods_per_year: int = 12
) -> pd.DataFrame:
    """
    Compare multiple models' predictions.
    
    Args:
        predictions: Dictionary of model_name -> predictions
        y_true: Actual values
        factor_names: Names of factors
        returns: Actual returns for investment metrics
        periods_per_year: Number of periods in a year
        
    Returns:
        DataFrame with comparison metrics for each model
    """
    comparison = {}
    
    for model_name, y_pred in predictions.items():
        metrics = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            factor_names=factor_names,
            returns=returns,
            periods_per_year=periods_per_year
        )
        
        comparison[model_name] = {
            'rmse': metrics.get('rmse', np.nan),
            'mae': metrics.get('mae', np.nan),
            'sharpe': metrics.get('investment', {}).get('sharpe_ratio', np.nan),
            'ann_return': metrics.get('investment', {}).get('annualized_return', np.nan),
            'volatility': metrics.get('investment', {}).get('annualized_volatility', np.nan),
            'max_drawdown': metrics.get('investment', {}).get('maximum_drawdown', np.nan),
            'calmar': metrics.get('investment', {}).get('calmar_ratio', np.nan),
            'win_rate': metrics.get('investment', {}).get('win_rate', np.nan),
        }
    
    df = pd.DataFrame(comparison).T
    df.index.name = 'model'
    
    return df


def format_metrics(metrics: Dict, decimals: int = 4) -> str:
    """
    Format metrics dictionary as readable string.
    
    Args:
        metrics: Dictionary of metrics
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    
    # Overall metrics
    if 'rmse' in metrics:
        lines.append(f"RMSE: {metrics['rmse']:.{decimals}f}")
    if 'mae' in metrics:
        lines.append(f"MAE: {metrics['mae']:.{decimals}f}")
    
    # Per-factor metrics
    if 'by_factor' in metrics:
        lines.append("\nPer-factor:")
        for factor, factor_metrics in metrics['by_factor'].items():
            lines.append(f"  {factor}: RMSE={factor_metrics['rmse']:.{decimals}f}, MAE={factor_metrics['mae']:.{decimals}f}")
    
    # Investment metrics
    if 'investment' in metrics:
        lines.append("\nInvestment:")
        inv = metrics['investment']
        lines.append(f"  Annualized Return: {inv.get('annualized_return', np.nan):.{decimals}f}%")
        lines.append(f"  Annualized Volatility: {inv.get('annualized_volatility', np.nan):.{decimals}f}%")
        lines.append(f"  Sharpe Ratio: {inv.get('sharpe_ratio', np.nan):.{decimals}f}")
        lines.append(f"  Maximum Drawdown: {inv.get('maximum_drawdown', np.nan):.{decimals}f}%")
        lines.append(f"  Calmar Ratio: {inv.get('calmar_ratio', np.nan):.{decimals}f}")
        lines.append(f"  Win Rate: {inv.get('win_rate', np.nan):.{decimals}f}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("Testing evaluation metrics...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    n_factors = 5
    
    y_true = np.random.randn(n_samples, n_factors)
    y_pred = y_true + np.random.randn(n_samples, n_factors) * 0.5
    
    returns = np.random.randn(n_samples) * 0.02  # Monthly returns (2% volatility)
    
    factor_names = ['Value', 'Momentum', 'Quality', 'LowVol', 'Size']
    
    # Test evaluate_predictions
    metrics = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        factor_names=factor_names,
        returns=returns,
        periods_per_year=12
    )
    
    print("\nMetrics:")
    print(format_metrics(metrics))
    
    # Test compare_models
    predictions = {
        'model_a': y_pred,
        'model_b': y_pred + np.random.randn(n_samples, n_factors) * 0.1,
    }
    
    comparison_df = compare_models(
        predictions=predictions,
        y_true=y_true,
        factor_names=factor_names,
        returns=returns
    )
    
    print("\nModel Comparison:")
    print(comparison_df.round(4))
    
    print("\nAll metrics tested successfully!")