"""
Utility functions for PyTorch MoE.
Data preparation, sequence creation, and evaluation helpers.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data with sequence windows.
    
    Creates sequences of length `seq_length` as inputs, with the next
    time step as the target.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_length: int = 12
    ):
        """
        Initialize dataset.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target matrix (n_samples, n_outputs)
            seq_length: Number of time steps in each sequence
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_length = seq_length
        
        # Validate
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have same length")
        if len(self.X) <= seq_length:
            raise ValueError(f"Not enough data: {len(self.X)} <= {seq_length}")
    
    def __len__(self) -> int:
        return len(self.X) - self.seq_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sequence and target.
        
        Returns:
            X_seq: (seq_length, n_features) - input sequence
            y_target: (n_outputs,) - target for next time step
        """
        X_seq = self.X[idx:idx + self.seq_length]
        y_target = self.y[idx + self.seq_length]
        return X_seq, y_target


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from time series data.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target matrix (n_samples, n_outputs)
        seq_length: Number of time steps in each sequence
        
    Returns:
        X_seq: (n_samples - seq_length, seq_length, n_features)
        y_seq: (n_samples - seq_length, n_outputs)
    """
    n_samples = len(X)
    n_features = X.shape[1]
    n_outputs = y.shape[1]
    
    X_seq = np.zeros((n_samples - seq_length, seq_length, n_features))
    y_seq = np.zeros((n_samples - seq_length, n_outputs))
    
    for i in range(n_samples - seq_length):
        X_seq[i] = X[i:i + seq_length]
        y_seq[i] = y[i + seq_length]
    
    return X_seq, y_seq


def prepare_moe_data(
    returns_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    seq_length: int = 12,
    lags: List[int] = [1, 3, 6, 12]
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Prepare data for PyTorch MoE training/evaluation.
    
    Args:
        returns_df: DataFrame of factor returns (monthly)
        macro_df: DataFrame of macro indicators (optional)
        seq_length: Number of time steps for LSTM sequences
        lags: Lag periods for feature creation
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target matrix (n_samples, n_outputs)
        dates: DatetimeIndex of aligned dates
    """
    from src.backtest import create_features
    
    # Create features using existing backtest logic
    X = create_features(returns_df, macro_df, lags)
    
    # Target: next month's returns
    y = returns_df.shift(-1).dropna()
    
    # Align X and y
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # Convert to numpy
    X_array = X.values.astype(np.float32)
    y_array = y.values.astype(np.float32)
    
    logger.info(f"Data prepared: {len(X_array)} samples, {X_array.shape[1]} features, {y_array.shape[1]} outputs")
    
    return X_array, y_array, common_idx


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets (time-ordered).
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target matrix (n_samples, n_outputs)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    logger.info(f"Split: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seq_length: int = 12,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and test.
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Data arrays
        seq_length: Sequence length for LSTM
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader