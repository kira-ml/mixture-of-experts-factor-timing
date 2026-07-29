"""
FRED macroeconomic data loader.
One class, fetch multiple series, apply transformations, align to factor dates.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fredapi import Fred
from typing import List, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FredLoader:
    """
    Load FRED macroeconomic data and prepare for factor timing pipeline.
    
    Usage:
        loader = FredLoader()
        fred_df = loader.load_series(
            start_date='2013-08-01',
            end_date='2026-07-31',
            series_ids=['CPIAUCSL', 'INDPRO', 'UNRATE', 'T10Y2Y', 'GS10', 'GS2']
        )
        # Returns DataFrame with all series aligned to month-end
    """
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        
        if not self.api_key:
            raise ValueError("FRED_API_KEY not found. Set in .env file.")
        
        self.fred = Fred(api_key=self.api_key)
        self.cache_dir = Path('data/raw/fred')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_series(
        self,
        start_date: str,
        end_date: str,
        series_ids: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch FRED series and return aligned DataFrame.
        
        Default series if None:
            - CPIAUCSL: Consumer Price Index (inflation)
            - INDPRO: Industrial Production
            - UNRATE: Unemployment Rate
            - T10Y2Y: 10-Year - 2-Year Treasury Spread
            - GS10: 10-Year Treasury Rate
            - GS2: 2-Year Treasury Rate
        """
        if series_ids is None:
            series_ids = ['CPIAUCSL', 'INDPRO', 'UNRATE', 'T10Y2Y', 'GS10', 'GS2']
        
        data = {}
        
        # Fetch with buffer to ensure we have data before the start date
        buffer_start = pd.to_datetime(start_date) - pd.DateOffset(years=2)
        buffer_start_str = buffer_start.strftime('%Y-%m-%d')
        
        for series_id in series_ids:
            try:
                series = self.fred.get_series(
                    series_id,
                    observation_start=buffer_start_str,
                    observation_end=end_date
                )
                data[series_id] = series
                logger.info(f"Fetched {series_id}: {len(series)} observations")
            except Exception as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")
        
        if not data:
            raise ValueError("No FRED data fetched. Check API key and series IDs.")
        
        # Combine into DataFrame
        df = pd.DataFrame(data)
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        # Resample to month-end (some series may be daily)
        if df.index.freq is None:
            df = df.resample('ME').last()
        elif df.index.freq.name != 'ME':
            df = df.resample('ME').last()
        
        # Forward-fill missing values (some series may start later)
        df = df.ffill()
        
        # Drop rows that are completely NaN (series that never started)
        df = df.dropna(how='all')
        
        # Ensure we have data in the requested range
        df = df.loc[start_date:end_date]
        
        # Drop any rows that still have NaN (unlikely after ffill)
        df = df.dropna()
        
        logger.info(f"Loaded {len(df)} months of FRED data from {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def add_transformations(
        self,
        df: pd.DataFrame,
        add_pct_change: bool = True,
        add_yoy: bool = True,
        add_zscore: bool = False
    ) -> pd.DataFrame:
        """
        Add transformed versions of FRED series.
        
        Returns DataFrame with original + transformed columns.
        """
        result = df.copy()
        
        for col in df.columns:
            # Month-over-month percent change
            if add_pct_change:
                result[f'{col}_pct_change'] = df[col].pct_change() * 100
            
            # Year-over-year percent change
            if add_yoy:
                result[f'{col}_yoy'] = df[col].pct_change(12) * 100
            
            # Z-score (standardized)
            if add_zscore:
                result[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
        
        return result
    
    def save(self, df: pd.DataFrame, filename: str = 'fred_processed.csv') -> Path:
        """Save processed FRED data to cache."""
        path = self.cache_dir / filename
        df.to_csv(path)
        logger.info(f"Saved FRED data to {path}")
        return path
    
    def load_cached(self, filename: str = 'fred_processed.csv') -> Optional[pd.DataFrame]:
        """Load cached FRED data if available."""
        path = self.cache_dir / filename
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            logger.info(f"Loaded cached FRED data: {len(df)} rows")
            return df
        return None


def load_fred_data(
    start_date: str = '2013-08-01',
    end_date: Optional[str] = None,
    series_ids: Optional[List[str]] = None,
    use_cache: bool = True,
    add_transforms: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load FRED data in one call.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        series_ids: List of FRED series IDs
        use_cache: Load from cache if available
        add_transforms: Add pct_change and yoy columns
        
    Returns:
        DataFrame with FRED data aligned to month-end
    """
    from datetime import datetime
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    loader = FredLoader()
    
    # Try cache
    if use_cache:
        df = loader.load_cached()
        if df is not None:
            return df
    
    # Fetch fresh
    df = loader.load_series(start_date, end_date, series_ids)
    
    # Add transformations
    if add_transforms:
        df = loader.add_transformations(df)
    
    # Save cache
    loader.save(df)
    
    return df


if __name__ == "__main__":
    # Quick test
    df = load_fred_data(start_date='2020-01-01', end_date='2023-12-31')
    print(f"\nFRED data shape: {df.shape}")
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")