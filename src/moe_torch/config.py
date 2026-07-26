"""
Configuration for PyTorch Mixture of Experts.
All hyperparameters centralized for easy experimentation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MoEConfig:
    """
    Configuration for PyTorch MoE model.
    
    Attributes:
        n_experts: Number of expert networks (regimes)
        input_size: Number of input features
        output_size: Number of output factors (6)
        hidden_size: Hidden dimension for LSTM and MLP experts
        lstm_layers: Number of LSTM layers in gating network
        expert_hidden_layers: Number of hidden layers in each expert MLP
        dropout: Dropout rate for regularization
        learning_rate: Learning rate for Adam optimizer
        weight_decay: L2 regularization strength
        epochs: Number of training epochs
        batch_size: Batch size for training
        patience: Early stopping patience (epochs)
        min_delta: Minimum change for early stopping
        seed: Random seed for reproducibility
    """
    
    # Model architecture
    n_experts: int = 4
    input_size: int = 28  # Will be auto-detected from data
    output_size: int = 6  # 6 factors (SPY, IWD, MTUM, QUAL, USMV, VIX)
    hidden_size: int = 32
    lstm_layers: int = 1
    expert_hidden_layers: int = 1
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 200
    batch_size: int = 32
    patience: int = 20
    min_delta: float = 1e-4
    seed: int = 42
    
    # Data
    sequence_length: int = 12  # Months of history for LSTM
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_experts < 2:
            raise ValueError("n_experts must be at least 2")
        if self.hidden_size < 8:
            raise ValueError("hidden_size must be at least 8")
        if self.epochs < 10:
            raise ValueError("epochs must be at least 10")