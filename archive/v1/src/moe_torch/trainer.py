"""
Training loop for PyTorch Mixture of Experts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import logging
from pathlib import Path

from .model import TorchMoE
from .config import MoEConfig

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(
    model: TorchMoE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred, gating_probs = model(X_batch)
        
        # Calculate loss (MSE on predictions)
        loss = criterion(pred, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate_epoch(
    model: TorchMoE,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate for one epoch.
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred, _ = model(X_batch)
            loss = criterion(pred, y_batch)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def train_moe(
    model: TorchMoE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: MoEConfig,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict:
    """
    Train the PyTorch MoE model.
    
    Args:
        model: TorchMoE model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: MoEConfig with hyperparameters
        device: torch device (auto-detects if None)
        verbose: Print progress
        
    Returns:
        Dictionary with training history and best model state
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    logger.info(f"Training on device: {device}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    iterator = range(config.epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training MoE")
    
    for epoch in iterator:
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        
        # Validate (only if val_loader is provided)
        if val_loader is not None:
            val_loss = validate_epoch(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            # Early stopping
            if early_stopping(val_loss):
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Progress
            current_lr = optimizer.param_groups[0]['lr']
            if verbose and epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"LR: {current_lr:.2e}"
                )
        else:
            # No validation - use training loss for best model tracking
            history['val_loss'].append(train_loss)  # Store train loss as placeholder
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_model_state = model.state_dict().copy()
            
            # Progress (less verbose when no validation)
            if verbose and epoch % 20 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs} - "
                    f"Train Loss: {train_loss:.4f}"
                )
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history['best_val_loss'] = best_val_loss
    history['best_epoch'] = len(history['train_loss'])
    
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    
    return history


def evaluate_moe(
    model: TorchMoE,
    test_loader: DataLoader,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Evaluate trained MoE model on test data.
    
    Args:
        model: Trained TorchMoE model
        test_loader: Test DataLoader
        device: torch device
        
    Returns:
        Dictionary with predictions, actuals, and metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_actuals = []
    all_gating_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred, gating_probs = model(X_batch)
            
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())
            all_gating_probs.append(gating_probs.cpu().numpy())
    
    predictions = np.vstack(all_preds)
    actuals = np.vstack(all_actuals)
    gating_probs = np.vstack(all_gating_probs)
    
    # Calculate metrics using existing evaluation module
    from src.evaluation import evaluate_predictions
    
    metrics = evaluate_predictions(
        y_true=actuals,
        y_pred=predictions,
        returns=None,
        periods_per_year=12
    )
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'gating_probs': gating_probs,
        'metrics': metrics
    }


def save_model(
    model: TorchMoE,
    config: MoEConfig,
    history: Dict,
    save_path: Path
) -> None:
    """
    Save trained model, config, and history.
    
    Args:
        model: Trained TorchMoE model
        config: MoEConfig used
        history: Training history
        save_path: Directory to save files
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), save_path / 'model_state.pt')
    
    # Save config
    import json
    with open(save_path / 'config.json', 'w') as f:
        json.dump(vars(config), f, indent=4, default=str)
    
    # Save history
    import pickle
    with open(save_path / 'history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    logger.info(f"Model saved to {save_path}")


def load_model(
    model: TorchMoE,
    load_path: Path,
    device: Optional[torch.device] = None
) -> TorchMoE:
    """
    Load trained model from disk.
    
    Args:
        model: TorchMoE instance with same architecture
        load_path: Directory containing model_state.pt
        device: torch device
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    state_dict = torch.load(load_path / 'model_state.pt', map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    logger.info(f"Model loaded from {load_path}")
    return model