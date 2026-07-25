"""
Data pipeline for downloading, cleaning, aligning, and saving factor data.
Week 1 MVP: Uses yfinance for factor proxies and VIX as macro proxy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Simple data pipeline for downloading and preparing factor data.
    
    Attributes:
        tickers (dict): Mapping of factor names to yfinance tickers
        start_date (str): Data download start date (YYYY-MM-DD)
        end_date (str): Data download end date (YYYY-MM-DD)
        data_dir (Path): Directory to save processed data
    """
    
    def __init__(self, start_date="1990-01-01", end_date=None):
        """
        Initialize the data pipeline.
        
        Args:
            start_date: Start date for data download (default: 1990-01-01)
            end_date: End date for data download (default: today)
        """
        self.tickers = {
            'SPY': 'SPY',      # Market (S&P 500 proxy)
            'IWD': 'IWD',      # Value (Russell 1000 Value)
            'MTUM': 'MTUM',    # Momentum (iShares MSCI USA Momentum)
            'QUAL': 'QUAL',    # Quality (iShares MSCI USA Quality)
            'USMV': 'USMV',    # Low Volatility (iShares MSCI USA Min Vol)
            'VIX': '^VIX',     # Macro proxy (Volatility Index)
        }
        
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        
        # Setup directories
        self.data_dir = Path('data')
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self._create_directories()
        
        logger.info(f"DataPipeline initialized: {self.start_date} to {self.end_date}")
        logger.info(f"Tickers: {list(self.tickers.keys())}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_data(self, save_raw=True):
        """
        Download all ticker data from yfinance.
        
        Args:
            save_raw: Whether to save raw downloaded data to CSV
            
        Returns:
            dict: Dictionary of DataFrames for each ticker
        """
        logger.info("Starting data download...")
        data = {}
        
        for name, ticker in self.tickers.items():
            logger.info(f"Downloading {name} ({ticker})...")
            try:
                # Download data
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                if df.empty:
                    logger.warning(f"No data for {name} ({ticker})")
                    continue
                
                data[name] = df
                logger.info(f"Downloaded {name}: {len(df)} rows")
                
                # Save raw data
                if save_raw:
                    raw_path = self.raw_dir / f"{name}_raw.csv"
                    df.to_csv(raw_path)
                    logger.info(f"Saved raw data to {raw_path}")
                    
            except Exception as e:
                logger.error(f"Error downloading {name} ({ticker}): {e}")
        
        logger.info(f"Download complete. Downloaded {len(data)} of {len(self.tickers)} tickers")
        return data
    
    def _resample_to_monthly(self, df, price_col='Close'):
        """
        Resample daily data to monthly end-of-month prices.
        
        Args:
            df: Daily DataFrame with price data (may have MultiIndex columns)
            price_col: Column name for prices (will try to find appropriate column)
            
        Returns:
            DataFrame with monthly data (single column 'Close')
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Check if we have MultiIndex columns (from yfinance with multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            # For MultiIndex, we need to select the first level (Close, Open, etc.)
            # and then the ticker name
            logger.info(f"MultiIndex columns detected: {df.columns.tolist()[:3]}...")
            
            # Get the first ticker symbol (the column level 1)
            ticker = df.columns[0][1] if len(df.columns[0]) > 1 else None
            
            if ticker:
                # Try to get the Close column for this ticker
                try:
                    # Select the Close column for this ticker
                    close_col = (price_col, ticker)
                    if close_col in df.columns:
                        use_col = close_col
                    else:
                        # Try to find any Close column
                        for col in df.columns:
                            if col[0] == price_col:
                                use_col = col
                                break
                        else:
                            # Use the first column
                            use_col = df.columns[0]
                            logger.warning(f"No Close column found. Using '{use_col}'")
                except Exception as e:
                    logger.warning(f"Error accessing MultiIndex: {e}")
                    use_col = df.columns[0]
            else:
                use_col = df.columns[0]
        else:
            # Single index columns (original behavior)
            logger.info(f"Single index columns: {df.columns.tolist()}")
            
            # Find the appropriate price column
            price_columns = ['Close', 'Adj Close', 'adj close', 'Adjusted']
            
            if price_col in df.columns:
                use_col = price_col
            else:
                for col in price_columns:
                    if col in df.columns:
                        use_col = col
                        break
                else:
                    use_col = df.columns[-1]
                    logger.warning(f"No price column found. Using '{use_col}'")
        
        logger.info(f"Using column: {use_col}")
        
        # Resample to month-end
        monthly = df.resample('ME').last()
        
        # Check if column exists after resampling
        if use_col not in monthly.columns:
            logger.warning(f"Column '{use_col}' not found after resampling. Available: {monthly.columns.tolist()[:5]}")
            # Try to find a suitable column
            for col in monthly.columns:
                if isinstance(col, tuple) and col[0] == price_col:
                    use_col = col
                    break
                elif isinstance(col, str) and 'close' in col.lower():
                    use_col = col
                    break
            else:
                use_col = monthly.columns[0]
                logger.info(f"Using first column: '{use_col}'")
        
        # Drop rows with NaN
        monthly = monthly.dropna(subset=[use_col])
        
        # Keep only the price column and rename to 'Close'
        monthly = monthly[[use_col]]
        monthly.columns = ['Close']
        
        return monthly
    
    def _calculate_monthly_returns(self, monthly_prices):
        """
        Calculate monthly returns from monthly prices.
        
        Args:
            monthly_prices: DataFrame with monthly price data
            
        Returns:
            Series of monthly returns (percentage change)
        """
        returns = monthly_prices.pct_change() * 100  # Convert to percentage
        return returns
    
    def align_data(self, data_dict):
        """
        Align all tickers to common dates and calculate returns.
        
        Args:
            data_dict: Dictionary of DataFrames from download_data()
            
        Returns:
            DataFrame: Aligned monthly returns for all factors
            DataFrame: Aligned monthly prices for all factors
        """
        logger.info("Aligning data to common dates...")
        
        monthly_prices = {}
        monthly_returns = {}
        
        for name, df in data_dict.items():
            # Resample to monthly
            monthly = self._resample_to_monthly(df)
            monthly_prices[name] = monthly['Close']
            
            # Calculate returns
            returns = self._calculate_monthly_returns(monthly['Close'])
            monthly_returns[name] = returns
        
        # Combine all into single DataFrames
        prices_df = pd.DataFrame(monthly_prices)
        returns_df = pd.DataFrame(monthly_returns)
        
        # Drop rows with any NaN (ensure all factors available)
        original_len = len(returns_df)
        prices_df = prices_df.dropna()
        returns_df = returns_df.dropna()
        
        dropped = original_len - len(returns_df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing data")
        
        logger.info(f"Aligned data shape: {returns_df.shape}")
        logger.info(f"Date range: {returns_df.index[0]} to {returns_df.index[-1]}")
        
        return returns_df, prices_df
    
    def validate_data(self, returns_df, prices_df):
        """
        Perform basic sanity checks on the data.
        
        Args:
            returns_df: DataFrame of monthly returns
            prices_df: DataFrame of monthly prices
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Validating data...")
        checks_passed = True
        
        # Check 1: No missing values
        if returns_df.isnull().any().any():
            logger.warning("Missing values found in returns data")
            checks_passed = False
        
        if prices_df.isnull().any().any():
            logger.warning("Missing values found in prices data")
            checks_passed = False
        
        # Check 2: Minimum rows
        min_rows = 60  # At least 5 years of monthly data
        if len(returns_df) < min_rows:
            logger.warning(f"Only {len(returns_df)} rows of data (< {min_rows} recommended)")
            checks_passed = False
        
        # Check 3: No extreme outliers (returns > 100% in a month)
        max_return = returns_df.abs().max().max()
        if max_return > 100:
            logger.warning(f"Extreme returns detected: {max_return:.2f}%")
            # This might be valid for VIX, so don't fail automatically
        
        # Check 4: VIX should be positive
        if 'VIX' in prices_df.columns and (prices_df['VIX'] <= 0).any():
            logger.warning("Negative or zero VIX values found")
            checks_passed = False
        
        # Check 5: Data types
        if not all(pd.api.types.is_numeric_dtype(returns_df[col]) for col in returns_df.columns):
            logger.warning("Non-numeric columns in returns data")
            checks_passed = False
        
        if checks_passed:
            logger.info("✅ All validation checks passed")
        else:
            logger.warning("⚠️ Some validation checks failed - review warnings above")
        
        return checks_passed
    
    def save_processed_data(self, returns_df, prices_df):
        """
        Save processed data to CSV files.
        
        Args:
            returns_df: DataFrame of monthly returns
            prices_df: DataFrame of monthly prices
        """
        # Save returns
        returns_path = self.processed_dir / 'monthly_returns.csv'
        returns_df.to_csv(returns_path)
        logger.info(f"Saved returns data to {returns_path}")
        
        # Save prices
        prices_path = self.processed_dir / 'monthly_prices.csv'
        prices_df.to_csv(prices_path)
        logger.info(f"Saved prices data to {prices_path}")
        
        # Save metadata
        metadata = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'n_months': len(returns_df),
            'n_factors': len(returns_df.columns),
            'factors': list(returns_df.columns),
            'date_range': f"{returns_df.index[0]} to {returns_df.index[-1]}"
        }
        
        metadata_path = self.processed_dir / 'metadata.csv'
        pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def run_pipeline(self):
        """
        Execute the complete data pipeline.
        
        Returns:
            tuple: (returns_df, prices_df) processed DataFrames
        """
        logger.info("=" * 50)
        logger.info("Starting data pipeline...")
        logger.info("=" * 50)
        
        # Step 1: Download data
        raw_data = self.download_data(save_raw=True)
        
        if not raw_data:
            logger.error("No data downloaded. Pipeline failed.")
            return None, None
        
        # Step 2: Align data
        returns_df, prices_df = self.align_data(raw_data)
        
        # Step 3: Validate
        self.validate_data(returns_df, prices_df)
        
        # Step 4: Save
        self.save_processed_data(returns_df, prices_df)
        
        logger.info("=" * 50)
        logger.info("Data pipeline complete!")
        logger.info("=" * 50)
        
        return returns_df, prices_df


def load_processed_data():
    """
    Load previously processed data from CSV files.
    
    Returns:
        tuple: (returns_df, prices_df) processed DataFrames
    """
    data_dir = Path('data/processed')
    
    returns_path = data_dir / 'monthly_returns.csv'
    prices_path = data_dir / 'monthly_prices.csv'
    
    if not returns_path.exists() or not prices_path.exists():
        logger.error("Processed data not found. Run the pipeline first.")
        return None, None
    
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    
    logger.info(f"Loaded returns data: {returns_df.shape}")
    logger.info(f"Loaded prices data: {prices_df.shape}")
    
    return returns_df, prices_df


if __name__ == "__main__":
    # Example usage when run as script
    pipeline = DataPipeline()
    returns, prices = pipeline.run_pipeline()
    
    if returns is not None:
        print("\nSample returns data:")
        print(returns.head())
        print(f"\nSummary statistics:")
        print(returns.describe())