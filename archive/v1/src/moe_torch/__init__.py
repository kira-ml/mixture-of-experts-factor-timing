"""
PyTorch Mixture of Experts module.
Isolated from the main pipeline for experimentation.
"""

from .model import TorchMoE, Expert, LSTMGatingNetwork
from .trainer import train_moe, evaluate_moe, save_model, load_model
from .config import MoEConfig
from .utils import (
    TimeSeriesDataset,
    create_sequences,
    prepare_moe_data,
    split_data,
    create_dataloaders
)

__all__ = [
    # Model
    'TorchMoE',
    'Expert',
    'LSTMGatingNetwork',
    
    # Training
    'train_moe',
    'evaluate_moe',
    'save_model',
    'load_model',
    
    # Config
    'MoEConfig',
    
    # Utils
    'TimeSeriesDataset',
    'create_sequences',
    'prepare_moe_data',
    'split_data',
    'create_dataloaders',
]

__version__ = '0.1.0'