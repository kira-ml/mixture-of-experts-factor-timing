"""
Backtesting framework for factor return prediction.
Week 1 MVP: Expanding window backtest with simple train-predict loop.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import logging
from datetime import datetime

from src.models import create_model
from src.evaluation import evaluate_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_features(
    returns_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    lags: List[int] = [1, 3, 6, 12]
) -> pd.DataFrame:
    """
    Create feature matrix from historical returns and macro indicators.
    
    Args:
        returns_df: DataFrame of factor returns (monthly)
        macro_df: DataFrame of macro indicators (e.g., VIX)
        lags: List of lag periods to include as features
        
    Returns:
        DataFrame with lagged features
    """
    features = []
    
    # Lagged factor returns
    for lag in lags:
        lagged = returns_df.shift(lag)
        lagged.columns = [f"{col}_lag_{lag}" for col in lagged.columns]
        features.append(lagged)
    
    # Macro indicators (if provided)
    if macro_df is not None:
        for lag in lags:
            lagged = macro_df.shift(lag)
            lagged.columns = [f"{col}_lag_{lag}" for col in lagged.columns]
            features.append(lagged)
    
    # Combine all features
    X = pd.concat(features, axis=1)
    
    # Drop rows with NaN (from shifting)
    X = X.dropna()
    
    return X


def prepare_backtest_data(
    returns_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    min_train_size: int = 60,
    test_size: int = 1,
    lags: List[int] = [1, 3, 6, 12]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp]]:
    """
    Prepare data for backtesting with expanding window.
    
    Args:
        returns_df: DataFrame of factor returns (monthly)
        macro_df: DataFrame of macro indicators
        min_train_size: Minimum number of months for training
        test_size: Number of months to predict (1 for monthly)
        lags: Lag periods for features
        
    Returns:
        X: Feature matrix
        y: Target matrix (next month returns)
        dates: List of dates for each observation
    """
    # Create features
    X = create_features(returns_df, macro_df, lags)
    
    # Target: next month's returns
    y = returns_df.shift(-1).dropna()
    
    # Align X and y
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    dates = X.index.tolist()
    
    logger.info(f"Backtest data prepared: {len(X)} observations")
    logger.info(f"Features: {X.shape[1]}, Factors: {y.shape[1]}")
    
    return X, y, dates


def expanding_window_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    dates: List[pd.Timestamp],
    min_train_size: int = 60,
    test_size: int = 1
) -> List[Dict[str, Union[pd.DataFrame, pd.Timestamp]]]:
    """
    Generate expanding window splits for backtesting.
    
    Args:
        X: Feature matrix
        y: Target matrix
        dates: List of dates
        min_train_size: Minimum training size (months)
        test_size: Test size (months)
        
    Returns:
        List of dictionaries with train/test indices and dates
    """
    n = len(X)
    splits = []
    
    # Ensure we have enough data for at least one split
    if n - min_train_size - test_size + 1 <= 0:
        logger.warning(f"Not enough data for backtest: n={n}, min_train={min_train_size}, test_size={test_size}")
        return splits
    
    for test_start in range(min_train_size, n - test_size + 1):
        # Train: from start to test_start - 1
        train_indices = list(range(0, test_start))
        # Test: test_start to test_start + test_size - 1
        test_indices = list(range(test_start, test_start + test_size))
        
        splits.append({
            'train_idx': train_indices,
            'test_idx': test_indices,
            'train_date': dates[0],
            'test_date': dates[test_start],
        })
    
    return splits



def predictions_to_returns(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    strategy: str = 'long_only_positive',
    cost_bps: float = 0.0
) -> np.ndarray:
    """
    Convert factor predictions to portfolio returns.
    
    Args:
        predictions: Predicted returns for each factor (n_samples, n_factors)
        actual_returns: Actual returns for each factor (n_samples, n_factors)
        strategy: Allocation strategy
            - 'long_only_positive': Equal-weight long positions on positive predictions
            - 'magnitude_weighted': Weight long positions by prediction magnitude
            - 'equal_weight': Equal-weight all factors (baseline)
        cost_bps: Transaction cost in basis points (e.g., 0.10 = 0.10%)
            
    Returns:
        Portfolio returns (n_samples,)
    """
    if strategy == 'long_only_positive':
        # For each month, allocate equally to factors with positive predicted return
        weights = np.where(predictions > 0, 1.0, 0.0)
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums != 0)
        
    elif strategy == 'magnitude_weighted':
        # Weight by prediction magnitude (only positive values)
        weights = np.where(predictions > 0, predictions, 0.0)
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums != 0)
        
    elif strategy == 'equal_weight':
        # Equal weight all factors (baseline for comparison)
        weights = np.ones_like(predictions) / predictions.shape[1]
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Portfolio return = sum(weights * actual_returns)
    portfolio_returns = np.sum(weights * actual_returns, axis=1)
    
    # Apply transaction costs
    if cost_bps > 0:
        # Cost as decimal (e.g., 0.10 bps = 0.0010%)
        cost_decimal = cost_bps / 10000
        
        # Calculate turnover: sum of absolute changes in weights
        prev_weights = np.zeros_like(weights)
        prev_weights[0] = 1.0 / weights.shape[1]
        
        turnover = np.zeros(len(weights))
        for i in range(1, len(weights)):
            turnover[i] = np.sum(np.abs(weights[i] - weights[i-1])) / 2
        
        transaction_costs = turnover * cost_decimal
        portfolio_returns = portfolio_returns - transaction_costs
    
    return portfolio_returns


def run_backtest(
    X: pd.DataFrame,
    y: pd.DataFrame,
    dates: List[pd.Timestamp],
    model_configs: List[Dict[str, Union[str, Dict]]],
    min_train_size: int = 60,
    test_size: int = 1,
    verbose: bool = True
) -> Dict[str, Dict[str, Union[pd.DataFrame, Dict]]]:
    """
    Run expanding window backtest for multiple models.
    
    Args:
        X: Feature matrix
        y: Target matrix
        dates: List of dates
        model_configs: List of model configurations
            Each config: {'name': str, 'params': dict}
        min_train_size: Minimum training size (months)
        test_size: Test size (months)
        verbose: Print progress
        
    Returns:
        Dictionary with results for each model:
            - predictions: DataFrame of predictions
            - actuals: DataFrame of actual values
            - dates: List of dates
            - metrics: Dictionary of evaluation metrics
            - fitted_model: Last fitted model (for MoE regime analysis)
    """
    n = len(X)
    n_factors = y.shape[1]
    factor_names = y.columns.tolist()
    
    # Store results
    results = {}
    
    # Create progress bar
    total_splits = n - min_train_size - test_size + 1
    iterator = tqdm(range(total_splits), desc="Backtesting") if verbose else range(total_splits)
    
    for model_config in model_configs:
        model_name = model_config['name']
        model_params = model_config.get('params', {})
        
        logger.info(f"Running backtest for {model_name}...")
        
        # Initialize storage for predictions and actuals
        all_predictions = []
        all_actuals = []
        all_dates = []
        last_fitted_model = None  # Store last fitted model for regime analysis
        
        # Get splits
        splits = expanding_window_split(X, y, dates, min_train_size, test_size)
        
        # Check if we have any splits
        if len(splits) == 0:
            logger.warning(f"No backtest splits available for {model_name}. Skipping.")
            results[model_name] = {
                'predictions': pd.DataFrame(columns=factor_names),
                'actuals': pd.DataFrame(columns=factor_names),
                'dates': [],
                'metrics': {},
                'fitted_model': None
            }
            continue
        
        for split_idx in iterator:
            train_idx = splits[split_idx]['train_idx']
            test_idx = splits[split_idx]['test_idx']
            test_date = splits[split_idx]['test_date']
            
            # Get train/test data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Create and train model
            model = create_model(model_name, **model_params)
            
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                # Store last fitted model for MoE
                if model_name == 'moe':
                    last_fitted_model = model
                
                # Store predictions
                all_predictions.append(pred)
                all_actuals.append(y_test.values)
                all_dates.append(test_date)
                
            except Exception as e:
                logger.error(f"Error with {model_name} at date {test_date}: {e}")
                # Fill with zeros if prediction fails
                all_predictions.append(np.zeros_like(y_test.values))
                all_actuals.append(y_test.values)
                all_dates.append(test_date)
        
        # Combine results
        predictions_df = pd.DataFrame(
            np.vstack(all_predictions),
            index=all_dates,
            columns=factor_names
        )
        
        # Debug: Check MoE predictions
        if model_name == 'moe':
            min_val = predictions_df.min().min()
            max_val = predictions_df.max().max()
            mean_val = predictions_df.mean().mean()
            pos_count = (predictions_df.values > 0).sum()
            total = predictions_df.size
            print(f"\n[DEBUG] MoE predictions: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
            print(f"[DEBUG] MoE positive predictions: {pos_count} out of {total} ({pos_count/total*100:.1f}%)\n")
        
        actuals_df = pd.DataFrame(
            np.vstack(all_actuals),
            index=all_dates,
            columns=factor_names
        )
        
        # Calculate portfolio returns from predictions
        portfolio_returns = predictions_to_returns(
            predictions=predictions_df.values,
            actual_returns=actuals_df.values,
            strategy='magnitude_weighted',  # Changed from 'long_only_positive'
            cost_bps=10.0
        )
        
        # Calculate metrics
        metrics = evaluate_predictions(
            y_true=actuals_df.values,
            y_pred=predictions_df.values,
            factor_names=factor_names,
            returns=portfolio_returns,
            periods_per_year=12
        )
        
        results[model_name] = {
            'predictions': predictions_df,
            'actuals': actuals_df,
            'dates': all_dates,
            'metrics': metrics,
            'fitted_model': last_fitted_model  # Store for regime analysis
        }
    
    return results


def backtest_models(
    returns_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    model_configs: Optional[List[Dict[str, Union[str, Dict]]]] = None,
    min_train_size: int = 60,
    test_size: int = 1,
    lags: List[int] = [1, 3, 6, 12],
    verbose: bool = True
) -> Dict[str, Dict[str, Union[pd.DataFrame, Dict]]]:
    """
    Main backtesting function for all models.
    
    Args:
        returns_df: DataFrame of factor returns (monthly)
        macro_df: DataFrame of macro indicators
        model_configs: List of model configurations
            If None, uses default models
        min_train_size: Minimum training size (months)
        test_size: Test size (months)
        lags: Lag periods for features
        verbose: Print progress
        
    Returns:
        Dictionary with results for each model
    """
    # Default models if not provided
    if model_configs is None:
        model_configs = [
            {'name': 'persistence', 'params': {}},
            {'name': 'rolling_avg', 'params': {'window': 12}},
            {'name': 'momentum', 'params': {'window': 12, 'decay': 0.9}},
            {'name': 'linear', 'params': {'standardize': True}},
            {'name': 'rf', 'params': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10}},
        ]
    
    # Prepare data
    X, y, dates = prepare_backtest_data(
        returns_df=returns_df,
        macro_df=macro_df,
        min_train_size=min_train_size,
        test_size=test_size,
        lags=lags
    )
    
    logger.info(f"Running backtest with {len(X)} observations, {X.shape[1]} features")
    logger.info(f"Models: {[config['name'] for config in model_configs]}")
    
    # Run backtest
    results = run_backtest(
        X=X,
        y=y,
        dates=dates,
        model_configs=model_configs,
        min_train_size=min_train_size,
        test_size=test_size,
        verbose=verbose
    )
    
    logger.info("Backtest complete!")
    
    return results


def summarize_results(
    results: Dict[str, Dict[str, Union[pd.DataFrame, Dict]]]
) -> pd.DataFrame:
    """
    Create summary DataFrame of backtest results.
    
    Args:
        results: Results from backtest_models()
        
    Returns:
        DataFrame with summary metrics for each model
    """
    summary = {}
    
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        
        summary[model_name] = {
            'rmse': metrics.get('rmse', np.nan),
            'mae': metrics.get('mae', np.nan),
            'sharpe': metrics.get('investment', {}).get('sharpe_ratio', np.nan),
            'ann_return': metrics.get('investment', {}).get('annualized_return', np.nan),
            'volatility': metrics.get('investment', {}).get('annualized_volatility', np.nan),
            'max_drawdown': metrics.get('investment', {}).get('maximum_drawdown', np.nan),
            'calmar': metrics.get('investment', {}).get('calmar_ratio', np.nan),
            'win_rate': metrics.get('investment', {}).get('win_rate', np.nan),
        }
    
    df = pd.DataFrame(summary).T
    df.index.name = 'model'
    
    return df


def get_predictions_for_model(
    results: Dict[str, Dict[str, Union[pd.DataFrame, Dict]]],
    model_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get predictions and actuals for a specific model.
    
    Args:
        results: Results from backtest_models()
        model_name: Name of the model
        
    Returns:
        Tuple of (predictions_df, actuals_df)
    """
    if model_name not in results:
        raise ValueError(f"Model {model_name} not found in results")
    
    model_results = results[model_name]
    return model_results['predictions'], model_results['actuals']


def get_best_model(
    results: Dict[str, Dict[str, Union[pd.DataFrame, Dict]]],
    metric: str = 'sharpe'
) -> Tuple[str, float]:
    """
    Get the best performing model based on a metric.
    
    Args:
        results: Results from backtest_models()
        metric: Metric to compare ('sharpe', 'calmar', 'rmse', etc.)
        
    Returns:
        Tuple of (model_name, metric_value)
    """
    best_model = None
    best_value = -np.inf if metric not in ['rmse', 'mae'] else np.inf
    
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        
        # Check if metric is in investment or top-level
        if metric in metrics.get('investment', {}):
            value = metrics['investment'][metric]
        elif metric in metrics:
            value = metrics[metric]
        else:
            continue
        
        if np.isnan(value):
            continue
        
        # Compare based on metric type
        if metric in ['rmse', 'mae']:
            # Lower is better
            if value < best_value:
                best_value = value
                best_model = model_name
        else:
            # Higher is better
            if value > best_value:
                best_value = value
                best_model = model_name
    
    return best_model, best_value


if __name__ == "__main__":
    # Example usage with dummy data
    print("Testing backtest framework...")
    
    # Create dummy data
    np.random.seed(42)
    n_months = 120
    n_factors = 5
    factor_names = ['Value', 'Momentum', 'Quality', 'LowVol', 'Size']
    
    # Generate returns with some structure
    returns = np.random.randn(n_months, n_factors) * 0.02
    returns_df = pd.DataFrame(returns, columns=factor_names)
    returns_df.index = pd.date_range('2010-01-01', periods=n_months, freq='M')
    
    # Create macro data (VIX)
    macro = np.random.randn(n_months, 1) * 0.1 + 20
    macro_df = pd.DataFrame(macro, columns=['VIX'])
    macro_df.index = returns_df.index
    
    # Test backtest
    results = backtest_models(
        returns_df=returns_df,
        macro_df=macro_df,
        min_train_size=36,
        test_size=1,
        lags=[1, 3, 6],
        verbose=True
    )
    
    # Summary
    summary = summarize_results(results)
    print("\nBacktest Summary:")
    print(summary.round(4))
    
    # Best model
    best_model, best_sharpe = get_best_model(results, 'sharpe')
    print(f"\nBest model by Sharpe: {best_model} ({best_sharpe:.4f})")
    
    print("\nBacktest framework tested successfully!")