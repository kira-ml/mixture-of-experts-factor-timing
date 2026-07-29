#!/usr/bin/env python
"""
Standalone runner for PyTorch MoE experiment.
Isolated from the main pipeline - uses existing data only.

Usage:
    # Single split (default)
    python run_moe_torch.py
    
    # Expanding window backtest (fair comparison with SimpleMoE)
    python run_moe_torch.py --expanding --min-train 60
    
    # Custom parameters
    python run_moe_torch.py --n-experts 4 --epochs 100 --batch-size 32
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

from src.data_pipeline import load_processed_data
from src.utils import set_seed, get_results_dir

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
    
    # Backtest options
    parser.add_argument(
        '--expanding',
        action='store_true',
        help='Run expanding window backtest instead of single split'
    )
    parser.add_argument(
        '--min-train',
        type=int,
        default=60,
        help='Minimum training size in months for expanding window (default: 60)'
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


    # Data parameters
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date for FRED data (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for FRED data (YYYY-MM-DD), defaults to today'
    )

    
    
    return parser.parse_args()


def run_single_split_experiment(args, returns_df, X, y, dates, device):
    """Run single train/val/test split experiment."""
    
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
    
    return test_results, model, config, history


def run_expanding_window_experiment(args, returns_df, X, y, dates, device):
    """Run expanding window backtest."""
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Expanding Window Backtest")
    logger.info("=" * 60)
    
    min_train_size = args.min_train
    n = len(X)
    n_splits = n - min_train_size - args.seq_length
    
    logger.info(f"Total samples: {n}")
    logger.info(f"Min train size: {min_train_size}")
    logger.info(f"Number of backtest windows: {n_splits}")
    
    # Storage for all predictions and actuals
    all_predictions = []
    all_actuals = []
    all_dates = []
    all_gating_probs = []
    
    from tqdm import tqdm
    
    for split_idx in tqdm(range(n_splits), desc="Expanding Window"):
        train_end = min_train_size + split_idx
        test_idx = train_end + args.seq_length
        
        if test_idx >= n:
            break
        
        # Split data
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_idx:test_idx + 1]
        y_test = y[test_idx:test_idx + 1]
        test_date = dates[test_idx]
        
        # Create training sequences
        from src.moe_torch.utils import create_sequences
        
        X_train_seq, y_train_seq = create_sequences(
            X_train, y_train, args.seq_length
        )
        
        from torch.utils.data import DataLoader, TensorDataset
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # Create model
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
        
        # Train model
        history = train_moe(
            model=model,
            train_loader=train_loader,
            val_loader=None,  # No validation for speed
            config=config,
            device=device,
            verbose=False
        )
        
        # Prepare test sequence (last seq_length months before test)
        X_test_seq = X[train_end:test_idx]
        if len(X_test_seq) < args.seq_length:
            continue
        
        X_test_seq = X_test_seq[-args.seq_length:].reshape(1, args.seq_length, -1)
        X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
        
        model.eval()
        with torch.no_grad():
            pred, gating_probs = model(X_test_tensor)
        
        # Store results
        all_predictions.append(pred.cpu().numpy().flatten())
        all_actuals.append(y_test.flatten())
        all_dates.append(test_date)
        all_gating_probs.append(gating_probs.cpu().numpy().flatten())
    
    # Combine results
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    gating_probs = np.array(all_gating_probs)
    test_dates = pd.DatetimeIndex(all_dates)
    
    logger.info(f"Backtest complete: {len(predictions)} predictions")
    
    # Return results in same format as single split
    test_results = {
        'predictions': predictions,
        'actuals': actuals,
        'gating_probs': gating_probs,
        'metrics': None  # Will be calculated later
    }
    
    return test_results, None, None, None, test_dates


def calculate_and_display_results(test_results, returns_df, args, is_expanding=False, test_dates=None):
    """Calculate metrics and display results."""
    
    # Get predictions and actuals
    predictions = test_results['predictions']
    actuals = test_results['actuals']
    
    # Calculate portfolio returns
    from src.backtest import predictions_to_returns
    
    portfolio_returns = predictions_to_returns(
        predictions=predictions,
        actual_returns=actuals,
        strategy='magnitude_weighted',
        cost_bps=10.0
    )
    
    # Calculate metrics
    from src.evaluation import (
        evaluate_predictions,
        sharpe_ratio,
        annualized_return,
        annualized_volatility,
        maximum_drawdown,
        calmar_ratio,
        win_rate
    )
    
    factor_names = returns_df.columns.tolist()
    metrics = evaluate_predictions(
        y_true=actuals,
        y_pred=predictions,
        factor_names=factor_names,
        returns=portfolio_returns,
        periods_per_year=12
    )
    
    inv_metrics = {
        'sharpe_ratio': sharpe_ratio(portfolio_returns),
        'annualized_return': annualized_return(portfolio_returns),
        'annualized_volatility': annualized_volatility(portfolio_returns),
        'maximum_drawdown': maximum_drawdown(portfolio_returns),
        'calmar_ratio': calmar_ratio(portfolio_returns),
        'win_rate': win_rate(portfolio_returns),
    }
    
    metrics['investment'] = inv_metrics
    test_results['metrics'] = metrics
    test_results['portfolio_returns'] = portfolio_returns
    test_results['investment_metrics'] = inv_metrics
    
    # Print results
    print("\n" + "=" * 80)
    if is_expanding:
        print("PYTORCH MOE EXPANDING WINDOW BACKTEST RESULTS")
        print(f"Predictions: {len(predictions)}")
        if test_dates is not None:
            print(f"Date range: {test_dates[0]} to {test_dates[-1]}")
    else:
        print("PYTORCH MOE SINGLE SPLIT RESULTS")
    print("=" * 80)
    
    print(f"\n[Predictive Metrics]")
    print(f"  RMSE: {metrics.get('rmse', np.nan):.4f}")
    print(f"  MAE:  {metrics.get('mae', np.nan):.4f}")
    
    if 'by_factor' in metrics:
        print("\n  Per-factor RMSE:")
        for factor, factor_metrics in metrics['by_factor'].items():
            print(f"    {factor}: {factor_metrics['rmse']:.4f}")
    
    print(f"\n[Investment Metrics] (Magnitude-Weighted, 10 bps)")
    print(f"  Sharpe Ratio:     {inv_metrics['sharpe_ratio']:.4f}")
    print(f"  Annual Return:    {inv_metrics['annualized_return']:.2f}%")
    print(f"  Volatility:       {inv_metrics['annualized_volatility']:.2f}%")
    print(f"  Max Drawdown:     {inv_metrics['maximum_drawdown']:.2f}%")
    print(f"  Calmar Ratio:     {inv_metrics['calmar_ratio']:.4f}")
    print(f"  Win Rate:         {inv_metrics['win_rate']:.2%}")
    
    print("\n" + "=" * 80)
    
    return metrics, inv_metrics, portfolio_returns


def save_results(test_results, returns_df, args, config, history, model, is_expanding=False, test_dates=None):
    """Save results to disk."""
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP: Saving Results")
    logger.info("=" * 60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.save and model is not None and config is not None:
        save_dir = get_results_dir() / f'torch_moe_{timestamp}'
        save_model(model, config, history, save_dir)
        logger.info(f"Model saved to {save_dir}")
    
    suffix = 'expanding' if is_expanding else 'singlesplit'
    results_dir = get_results_dir() / f'torch_moe_{suffix}_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    factor_names = returns_df.columns.tolist()
    if test_dates is not None:
        idx = test_dates
    else:
        idx = range(len(test_results['predictions']))
    
    pred_df = pd.DataFrame(
        test_results['predictions'],
        columns=factor_names,
        index=idx
    )
    pred_df.to_csv(results_dir / 'predictions.csv')
    
    actual_df = pd.DataFrame(
        test_results['actuals'],
        columns=factor_names,
        index=idx
    )
    actual_df.to_csv(results_dir / 'actuals.csv')
    
    # Save gating probabilities
    n_experts = test_results['gating_probs'].shape[1]
    gating_df = pd.DataFrame(
        test_results['gating_probs'],
        columns=[f'Regime_{i+1}' for i in range(n_experts)],
        index=idx
    )
    gating_df.to_csv(results_dir / 'gating_probs.csv')
    
    # Save portfolio returns
    portfolio_df = pd.DataFrame(
        test_results['portfolio_returns'],
        columns=['portfolio_returns'],
        index=idx
    )
    portfolio_df.to_csv(results_dir / 'portfolio_returns.csv')
    
    # Save combined metrics
    inv_metrics = test_results['investment_metrics']
    metrics = test_results['metrics']
    combined_metrics = {
        'rmse': metrics.get('rmse', np.nan),
        'mae': metrics.get('mae', np.nan),
        'sharpe_ratio': inv_metrics['sharpe_ratio'],
        'annualized_return': inv_metrics['annualized_return'],
        'annualized_volatility': inv_metrics['annualized_volatility'],
        'maximum_drawdown': inv_metrics['maximum_drawdown'],
        'calmar_ratio': inv_metrics['calmar_ratio'],
        'win_rate': inv_metrics['win_rate']
    }
    combined_df = pd.DataFrame([combined_metrics])
    combined_df.to_csv(results_dir / 'combined_metrics.csv', index=False)
    
    logger.info(f"Results saved to {results_dir}")


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
    if args.expanding:
        logger.info("PYTORCH MOE EXPANDING WINDOW BACKTEST")
    else:
        logger.info("PYTORCH MOE STANDALONE EXPERIMENT (SINGLE SPLIT)")
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
    
    macro_df = None
    
    # Try to load FRED data first
    try:
        from src.fred_data import load_fred_data
        macro_df = load_fred_data(
            start_date=args.start_date if hasattr(args, 'start_date') else '2020-01-01',
            end_date=args.end_date if hasattr(args, 'end_date') else None,
            use_cache=True,
            add_transforms=True
        )
        logger.info(f"Loaded FRED data: {macro_df.shape[1]} series, {len(macro_df)} months")
    except Exception as e:
        logger.warning(f"FRED data not available: {e}. Falling back to VIX.")
        if 'VIX' in returns_df.columns:
            macro_df = returns_df[['VIX']]
            logger.info("Using VIX as macro indicator")
        else:
            logger.info("No macro indicator found")



    
    lags = [1, 3, 6, 12]
    X, y, dates = prepare_moe_data(
        returns_df=returns_df,
        macro_df=macro_df,
        seq_length=args.seq_length,
        lags=lags
    )
    
    logger.info(f"Features: {X.shape}, Targets: {y.shape}")
    logger.info(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Step 3: Run experiment (single split or expanding window)
    if args.expanding:
        test_results, model, config, history, test_dates = run_expanding_window_experiment(
            args, returns_df, X, y, dates, device
        )
    else:
        test_results, model, config, history = run_single_split_experiment(
            args, returns_df, X, y, dates, device
        )
        test_dates = None
    
    # Step 4: Calculate and display results
    metrics, inv_metrics, portfolio_returns = calculate_and_display_results(
        test_results, returns_df, args, 
        is_expanding=args.expanding,
        test_dates=test_dates
    )
    
    # Step 5: Save results
    save_results(
        test_results, returns_df, args, config, history, model,
        is_expanding=args.expanding,
        test_dates=test_dates
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()