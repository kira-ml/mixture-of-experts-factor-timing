"""
Shared utility functions for the factor timing project.
Week 1 MVP: Simple helper functions for dates, validation, and configuration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import yaml
from typing import Union, List, Optional, Dict, Any, Tuple
import sys

# Configure logging (will be configured by main script)
logger = logging.getLogger(__name__)


# ============================================================================
# DATE HANDLING UTILITIES
# ============================================================================

def get_month_end_dates(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> List[datetime]:
    """
    Generate list of month-end dates between start and end.
    
    Args:
        start_date: Start date (YYYY-MM-DD or datetime)
        end_date: End date (YYYY-MM-DD or datetime)
        
    Returns:
        List of datetime objects (month-end dates)
    """
    # Convert to datetime if string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate month-end dates
    dates = []
    current = start_date.replace(day=1)
    end = end_date.replace(day=1)
    
    while current <= end:
        # Get last day of month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        month_end = next_month - timedelta(days=1)
        
        if month_end >= start_date and month_end <= end_date:
            dates.append(month_end)
        
        current = next_month
    
    return dates


def get_month_start_dates(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> List[datetime]:
    """
    Generate list of month-start dates between start and end.
    
    Args:
        start_date: Start date (YYYY-MM-DD or datetime)
        end_date: End date (YYYY-MM-DD or datetime)
        
    Returns:
        List of datetime objects (month-start dates)
    """
    # Convert to datetime if string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate month-start dates
    dates = []
    current = start_date.replace(day=1)
    end = end_date.replace(day=1)
    
    while current <= end:
        if current >= start_date:
            dates.append(current)
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)
    
    return dates


def to_monthly_period(date: Union[str, datetime, pd.Timestamp]) -> str:
    """
    Convert date to monthly period string (YYYY-MM).
    
    Args:
        date: Date to convert
        
    Returns:
        String in format YYYY-MM
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    
    return date.strftime('%Y-%m')


def parse_period(period: str) -> datetime:
    """
    Parse period string (YYYY-MM) to datetime (first day of month).
    
    Args:
        period: String in format YYYY-MM
        
    Returns:
        datetime object (first day of month)
    """
    return datetime.strptime(period, '%Y-%m')


# ============================================================================
# DATA VALIDATION UTILITIES
# ============================================================================

def check_data_consistency(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    expected_index_type: str = 'datetime',
    min_rows: int = 1,
    max_missing_pct: float = 0.05
) -> Tuple[bool, Dict[str, Any]]:
    """
    Perform basic consistency checks on a DataFrame.
    
    Args:
        df: DataFrame to check
        expected_columns: List of expected column names (optional)
        expected_index_type: Expected index type ('datetime', 'period', or None)
        min_rows: Minimum number of rows required
        max_missing_pct: Maximum percentage of missing values allowed
        
    Returns:
        Tuple of (passed: bool, details: dict)
    """
    details = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': df.columns.tolist(),
        'has_na': df.isnull().any().any(),
        'na_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) if len(df) > 0 else 1.0,
        'index_type': str(df.index),
        'issues': []
    }
    
    passed = True
    
    # Check 1: Minimum rows
    if len(df) < min_rows:
        details['issues'].append(f"Only {len(df)} rows (< {min_rows} required)")
        passed = False
    
    # Check 2: Expected columns
    if expected_columns is not None:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            details['issues'].append(f"Missing columns: {missing_cols}")
            passed = False
    
    # Check 3: Index type
    if expected_index_type == 'datetime':
        if not isinstance(df.index, pd.DatetimeIndex):
            details['issues'].append("Index is not DatetimeIndex")
            passed = False
    elif expected_index_type == 'period':
        if not isinstance(df.index, pd.PeriodIndex):
            details['issues'].append("Index is not PeriodIndex")
            passed = False
    
    # Check 4: Missing values
    if details['na_pct'] > max_missing_pct:
        details['issues'].append(f"Missing values: {details['na_pct']:.2%} (> {max_missing_pct:.2%})")
        passed = False
    
    # Check 5: Duplicate index
    if not df.index.is_unique:
        details['issues'].append("Duplicate index values found")
        passed = False
    
    # Check 6: Data types
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        details['issues'].append(f"Non-numeric columns: {non_numeric}")
        passed = False
    
    return passed, details


def validate_factor_returns(
    returns_df: pd.DataFrame,
    expected_factors: Optional[List[str]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Specialized validation for factor returns data.
    
    Args:
        returns_df: DataFrame of factor returns
        expected_factors: List of expected factor names
        
    Returns:
        Tuple of (passed: bool, details: dict)
    """
    if expected_factors is None:
        expected_factors = ['Value', 'Momentum', 'Quality', 'LowVol', 'Size']
    
    # Basic consistency check
    passed, details = check_data_consistency(
        returns_df,
        expected_columns=expected_factors,
        expected_index_type='datetime',
        min_rows=36
    )
    
    # Additional checks specific to returns
    if passed:
        # Check for extreme outliers (> 100% in a month)
        max_abs_return = returns_df.abs().max().max()
        if max_abs_return > 100:
            details['issues'].append(f"Extreme returns detected: {max_abs_return:.2f}%")
            details['max_abs_return'] = max_abs_return
            passed = False
    
    return passed, details


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def load_default_config() -> Dict[str, Any]:
    """
    Load default configuration for the project.
    
    Returns:
        Dictionary with default parameters
    """
    return {
        'data': {
            'start_date': '1990-01-01',
            'end_date': None,  # None means today
            'tickers': {
                'SPY': 'SPY',
                'IWD': 'IWD',
                'MTUM': 'MTUM',
                'QUAL': 'QUAL',
                'USMV': 'USMV',
                'VIX': '^VIX'
            }
        },
        'features': {
            'lags': [1, 3, 6, 12],
            'include_macro': True
        },
        'backtest': {
            'min_train_size': 60,  # 5 years
            'test_size': 1,  # 1 month
            'periods_per_year': 12
        },
        'models': {
            'rolling_avg': {'window': 12},
            'momentum': {'window': 12, 'decay': 0.9},
            'linear': {'standardize': True},
            'rf': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10}
        }
    }


def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to config file (.json or .yaml/.yml)
        
    Returns:
        Dictionary with configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4, default=str)
    
    logger.info(f"Config saved to {save_path}")


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Path to log file (optional)
        console: Whether to log to console
        
    Returns:
        Root logger
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


# ============================================================================
# FILE SYSTEM UTILITIES
# ============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get project root directory (where src/ is located).
    
    Returns:
        Path to project root
    """
    # Assuming this file is in src/
    current = Path(__file__).resolve()
    return current.parent.parent


def get_data_dir(subdir: Optional[str] = None) -> Path:
    """
    Get data directory path.
    
    Args:
        subdir: Optional subdirectory name ('raw', 'processed', 'results')
        
    Returns:
        Path to data directory
    """
    root = get_project_root()
    data_dir = root / 'data'
    
    if subdir:
        data_dir = data_dir / subdir
    
    ensure_directory(data_dir)
    return data_dir


def get_results_dir() -> Path:
    """
    Get results directory path.
    
    Returns:
        Path to results directory
    """
    root = get_project_root()
    results_dir = root / 'results'
    ensure_directory(results_dir)
    return results_dir


# ============================================================================
# DATA TRANSFORMATION UTILITIES
# ============================================================================

def winsorize(
    data: np.ndarray,
    limits: Tuple[float, float] = (0.01, 0.99)
) -> np.ndarray:
    """
    Winsorize data (cap outliers at specified percentiles).
    
    Args:
        data: Array of values
        limits: Tuple of (lower_percentile, upper_percentile)
        
    Returns:
        Winsorized array
    """
    data = np.array(data).copy()
    lower = np.percentile(data, limits[0] * 100)
    upper = np.percentile(data, limits[1] * 100)
    data[data < lower] = lower
    data[data > upper] = upper
    return data


def standardize(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = 'zscore'
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Standardize data using various methods.
    
    Args:
        data: Array or DataFrame to standardize
        method: 'zscore' (zero mean, unit variance) or 'minmax' (0 to 1)
        
    Returns:
        Standardized data (same type as input)
    """
    is_dataframe = isinstance(data, pd.DataFrame)
    
    if is_dataframe:
        values = data.values
    else:
        values = np.array(data)
    
    if method == 'zscore':
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std[std == 0] = 1
        standardized = (values - mean) / std
    elif method == 'minmax':
        min_val = np.nanmin(values, axis=0)
        max_val = np.nanmax(values, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        standardized = (values - min_val) / range_val
    else:
        raise ValueError(f"Unknown standardization method: {method}")
    
    if is_dataframe:
        return pd.DataFrame(standardized, index=data.index, columns=data.columns)
    else:
        return standardized


# ============================================================================
# TIME SERIES UTILITIES
# ============================================================================

def lag_series(
    data: pd.Series,
    lags: List[int],
    fill_na: bool = False
) -> pd.DataFrame:
    """
    Create lagged versions of a time series.
    
    Args:
        data: Time series
        lags: List of lag periods
        fill_na: Whether to fill NaN values (forward fill)
        
    Returns:
        DataFrame with lagged columns
    """
    lagged = {}
    for lag in lags:
        lagged[f"lag_{lag}"] = data.shift(lag)
    
    df = pd.DataFrame(lagged)
    
    if fill_na:
        df = df.ffill()
    
    return df


def rolling_metrics(
    returns: pd.Series,
    window: int = 12,
    metrics: List[str] = ['mean', 'std', 'sharpe']
) -> pd.DataFrame:
    """
    Calculate rolling metrics for returns.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        metrics: List of metrics to calculate
        
    Returns:
        DataFrame with rolling metrics
    """
    results = {}
    
    for metric in metrics:
        if metric == 'mean':
            results[metric] = returns.rolling(window).mean()
        elif metric == 'std':
            results[metric] = returns.rolling(window).std()
        elif metric == 'sharpe':
            # Simplified Sharpe (assuming zero risk-free rate)
            mean = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            results[metric] = mean / std * np.sqrt(12)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return pd.DataFrame(results)


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_number(
    value: float,
    decimals: int = 4,
    as_percentage: bool = False
) -> str:
    """
    Format a number for display.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        as_percentage: Display as percentage
        
    Returns:
        Formatted string
    """
    if np.isnan(value):
        return "NaN"
    
    if as_percentage:
        return f"{value * 100:.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}"


def format_currency(
    value: float,
    decimals: int = 2,
    currency: str = '$'
) -> str:
    """
    Format a number as currency.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        currency: Currency symbol
        
    Returns:
        Formatted string
    """
    if np.isnan(value):
        return "NaN"
    
    return f"{currency}{abs(value):,.{decimals}f}"


# ============================================================================
# SEED UTILITIES
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seed set to {seed}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test date utilities
    dates = get_month_end_dates('2020-01-01', '2020-12-31')
    print(f"Generated {len(dates)} month-end dates (2020)")
    
    # Test validation
    df = pd.DataFrame(
        np.random.randn(100, 5),
        columns=['Value', 'Momentum', 'Quality', 'LowVol', 'Size'],
        index=pd.date_range('2010-01-01', periods=100, freq='M')
    )
    passed, details = check_data_consistency(df)
    print(f"Data validation: {'Passed' if passed else 'Failed'}")
    
    # Test config
    config = load_default_config()
    print(f"Loaded config with {len(config)} sections")
    
    # Test project root
    root = get_project_root()
    print(f"Project root: {root}")
    
    print("\nAll utilities tested successfully!")