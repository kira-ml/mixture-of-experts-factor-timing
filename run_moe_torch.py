#!/usr/bin/env python
"""
Standalone runner for PyTorch MoE experiment.
Isolated from the main pipeline - uses existing data only.
Does not affect main.py, backtest.py, or existing moe.py.

Usage:
    python run_moe_torch.py
    python run_moe_torch.py --epochs 100 --batch-size 16
    python run_moe_torch.py --n-experts 3 --hidden-size 64
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_processed_data, set_seed, get_results_dir
from src.moe_torch import (
    TorchMoE,
    MoEConfig,
    train_moe,
    evaluate_moe,
    save_model,
    prepare_moe_data,
    split_data,
    create_dataloaders
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch MoE Standalone Runner (Isolated from main pipeline)"
    )
    
    # Model architecture
    parser.add_argument(
        '--n-experts',
        type=int,
        default=4,
        help='Number of experts/regimes (default: 4)'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=32,
        help='Hidden size for LSTM and experts (default: 32)'
    )
    parser.add_argument(
        '--lstm-layers',
        type=int,
        default=1,
        help='Number of LSTM layers (default: 1)'
    )
    parser.add_argument(
        '--expert-hidden-layers',
        type=int,
        default=1,
        help='Hidden layers per expert (default: 1)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate (default: 0.1)'
    )
    
    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Training epochs (default: 200)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='L2 regularization (default: 1e-5)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (default: 20)'
    )
    
    # Data
    parser.add_argument(
        '--seq-length',
        type=int,
        default=12,
        help='Sequence length for LSTM (default: 12)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training data ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation data ratio (default: 0.15)'
    )
    
    # General
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save trained model to disk'
    )
    
    return parser.parse_args()


def run_experiment(args):
    """Main experiment runner."""
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("=" * 60)
    logger.info("PYTORCH MOE STANDALONE EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Configuration: {vars(args)}")
    
    # Step 1: Load data from existing pipeline
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 60)
    
    returns_df, prices_df = load_processed_data()
    
    if returns_df is None:
        logger.error("No data found. Run main.py --download-data first.")
        return
    
    logger.info(f"Returns data: {returns_df.shape}")
    logger.info(f"Factors: {returns_df.columns.tolist()}")
    
    # Step 2: Prepare features
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Preparing Features")
    logger.info("=" * 60)
    
    # Extract VIX as macro indicator
    macro_df = None
    if 'VIX' in returns_df.columns:
        macro_df = returns_df[['VIX']]
        logger.info("Using VIX as macro indicator")
    else:
        logger.info("No macro indicator found")
    
    # Prepare data for MoE (using same lags as main pipeline)
    lags = [1, 3, 6, 12]
    X, y, dates = prepare_moe_data(
        returns_df=returns_df,
        macro_df=macro_df,
        seq_length=args.seq_length,
        lags=lags
    )
    
    logger.info(f"Features: {X.shape}, Targets: {y.shape}")
    logger.info(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Step 3: Split data (time-ordered)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Splitting Data (Time-Ordered)")
    logger.info("=" * 60)
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X=X,
        y=y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Step 4: Create DataLoaders
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Creating DataLoaders")
    logger.info("=" * 60)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        seq_length=args.seq_length,
        batch_size=args.batch_size
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Step 5: Create model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Creating Model")
    logger.info("=" * 60)
    
    config = MoEConfig(
        n_experts=args.n_experts,
        input_size=X.shape[1],
        output_size=y.shape[1],
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        expert_hidden_layers=args.expert_hidden_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        sequence_length=args.seq_length,
        seed=args.seed
    )
    
    model = TorchMoE(
        n_experts=config.n_experts,
        input_size=config.input_size,
        output_size=config.output_size,
        hidden_size=config.hidden_size,
        lstm_layers=config.lstm_layers,
        expert_hidden_layers=config.expert_hidden_layers,
        dropout=config.dropout
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Config: {config}")
    
    # Step 6: Train model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Training")
    logger.info("=" * 60)
    
    history = train_moe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        verbose=True
    )
    
    logger.info(f"Training complete. Best val loss: {history['best_val_loss']:.4f}")
    logger.info(f"Best epoch: {history['best_epoch']}")
    
    # Step 7: Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Evaluation on Test Set")
    logger.info("=" * 60)
    
    test_results = evaluate_moe(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    metrics = test_results['metrics']
    
    print("\n" + "=" * 80)
    print("PYTORCH MOE TEST RESULTS")
    print("=" * 80)
    print(f"RMSE: {metrics.get('rmse', np.nan):.4f}")
    print(f"MAE:  {metrics.get('mae', np.nan):.4f}")
    
    if 'by_factor' in metrics:
        print("\nPer-factor RMSE:")
        for factor, factor_metrics in metrics['by_factor'].items():
            print(f"  {factor}: {factor_metrics['rmse']:.4f}")
    
    print("\n" + "=" * 80)
    
    # Step 8: Save results
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Saving Results")
    logger.info("=" * 60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.save:
        save_dir = get_results_dir() / f'torch_moe_{timestamp}'
        save_model(model, config, history, save_dir)
        logger.info(f"Model saved to {save_dir}")
    
    # Save predictions
    results_dir = get_results_dir() / f'torch_moe_predictions_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(
        test_results['predictions'],
        columns=returns_df.columns.tolist()
    )
    pred_df.to_csv(results_dir / 'predictions.csv')
    
    actual_df = pd.DataFrame(
        test_results['actuals'],
        columns=returns_df.columns.tolist()
    )
    actual_df.to_csv(results_dir / 'actuals.csv')
    
    # Save gating probabilities
    gating_df = pd.DataFrame(
        test_results['gating_probs'],
        columns=[f'Regime_{i+1}' for i in range(config.n_experts)]
    )
    gating_df.to_csv(results_dir / 'gating_probs.csv')
    
    logger.info(f"Results saved to {results_dir}")
    
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()