"""
Main orchestrator for the factor timing project.
Week 1 MVP: Run data pipeline, backtest models, and generate summary results.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Import project modules
from src.data_pipeline import DataPipeline, load_processed_data
from src.backtest import backtest_models, summarize_results, get_best_model
from src.evaluation import format_metrics
from src.utils import setup_logging, set_seed, get_data_dir, get_results_dir, load_default_config

# Configure logging
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Factor Timing Project - Main Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python main.py --run-all
  
  # Only download and process data
  python main.py --download-data
  
  # Run backtest with existing data
  python main.py --backtest
  
  # Run with custom settings
  python main.py --run-all --start-date 2000-01-01 --min-train 84
        """
    )
    
    # Main actions
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run the complete pipeline (download + backtest)'
    )
    parser.add_argument(
        '--download-data',
        action='store_true',
        help='Only download and process data'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Only run backtest (requires existing data)'
    )
    
    # Data parameters
    parser.add_argument(
        '--start-date',
        type=str,
        default='1990-01-01',
        help='Start date for data download (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for data download (YYYY-MM-DD), defaults to today'
    )
    
    # Backtest parameters
    parser.add_argument(
        '--min-train',
        type=int,
        default=60,
        help='Minimum training window size in months'
    )
    parser.add_argument(
        '--lags',
        type=int,
        nargs='+',
        default=[1, 3, 6, 12],
        help='Lag periods for features'
    )
    
    # Model parameters
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['persistence', 'rolling_avg', 'momentum', 'linear', 'rf'],
        help='Models to run (persistence, rolling_avg, momentum, linear, rf)'
    )
    
    # General parameters
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--no-console-log',
        action='store_true',
        help='Disable console logging'
    )
    
    return parser.parse_args()


def build_model_configs(model_names: list) -> list:
    """
    Build model configurations from model names.
    
    Args:
        model_names: List of model names
        
    Returns:
        List of model configurations
    """
    model_configs = []
    
    # Default parameters for each model
    model_params = {
        'persistence': {},
        'rolling_avg': {'window': 12},
        'momentum': {'window': 12, 'decay': 0.9},
        'linear': {'standardize': True},
        'rf': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10},
    }
    
    for name in model_names:
        if name in model_params:
            model_configs.append({
                'name': name,
                'params': model_params[name].copy()
            })
        else:
            logger.warning(f"Unknown model: {name}, skipping")
    
    return model_configs


def run_data_pipeline(args) -> tuple:
    """
    Run the data pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (returns_df, prices_df)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Data Pipeline")
    logger.info("=" * 60)
    
    # Initialize pipeline
    pipeline = DataPipeline(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Run pipeline
    returns_df, prices_df = pipeline.run_pipeline()
    
    if returns_df is None:
        logger.error("Data pipeline failed. Exiting.")
        sys.exit(1)
    
    logger.info(f"Data pipeline complete: {len(returns_df)} months of data")
    logger.info(f"Factors: {returns_df.columns.tolist()}")
    
    return returns_df, prices_df


def run_backtest_pipeline(args, returns_df: pd.DataFrame) -> dict:
    """
    Run the backtest pipeline.
    
    Args:
        args: Command line arguments
        returns_df: DataFrame of factor returns
        
    Returns:
        Dictionary of backtest results
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Backtest")
    logger.info("=" * 60)
    
    # Extract VIX as macro indicator (if available)
    macro_df = None
    if 'VIX' in returns_df.columns:
        macro_df = returns_df[['VIX']]
        logger.info("Using VIX as macro indicator")
    else:
        logger.info("No macro indicator found (VIX column missing)")
    
    # Build model configurations
    model_configs = build_model_configs(args.models)
    logger.info(f"Models to evaluate: {[m['name'] for m in model_configs]}")
    
    # Run backtest
    results = backtest_models(
        returns_df=returns_df,
        macro_df=macro_df,
        model_configs=model_configs,
        min_train_size=args.min_train,
        test_size=1,
        lags=args.lags,
        verbose=True
    )
    
    return results


def display_results(results: dict) -> None:
    """
    Display backtest results in a readable format.
    
    Args:
        results: Backtest results dictionary
    """
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    
    # Generate summary DataFrame
    summary = summarize_results(results)
    
    # Display summary
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(summary.round(4).to_string())
    
    # Find best models
    print("\n" + "-" * 80)
    print("BEST PERFORMING MODELS")
    print("-" * 80)
    
    # Best by Sharpe
    best_sharpe_model, best_sharpe = get_best_model(results, 'sharpe')
    print(f"Best Sharpe Ratio:    {best_sharpe_model} ({best_sharpe:.4f})")
    
    # Best by Calmar
    best_calmar_model, best_calmar = get_best_model(results, 'calmar')
    print(f"Best Calmar Ratio:    {best_calmar_model} ({best_calmar:.4f})")
    
    # Best by RMSE (lower is better)
    best_rmse_model, best_rmse = get_best_model(results, 'rmse')
    print(f"Lowest RMSE:          {best_rmse_model} ({best_rmse:.4f})")
    
    # Best by Annual Return
    best_return_model, best_return = get_best_model(results, 'ann_return')
    print(f"Best Annual Return:   {best_return_model} ({best_return:.2f}%)")
    
    # Best by Win Rate
    best_win_model, best_win = get_best_model(results, 'win_rate')
    print(f"Best Win Rate:        {best_win_model} ({best_win:.2%})")
    
    print("\n" + "=" * 80)
    
    # Detailed per-model metrics
    print("\nDETAILED METRICS BY MODEL")
    print("-" * 80)
    
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        print(f"\n{model_name.upper()}:")
        
        # Predictive metrics
        print(f"  RMSE: {metrics.get('rmse', np.nan):.4f}")
        print(f"  MAE:  {metrics.get('mae', np.nan):.4f}")
        
        # Per-factor metrics
        if 'by_factor' in metrics:
            print("  Per-factor RMSE:")
            for factor, factor_metrics in metrics['by_factor'].items():
                print(f"    {factor}: {factor_metrics['rmse']:.4f}")
        
        # Investment metrics
        if 'investment' in metrics:
            inv = metrics['investment']
            print(f"  Sharpe:       {inv.get('sharpe_ratio', np.nan):.4f}")
            print(f"  Return:       {inv.get('annualized_return', np.nan):.2f}%")
            print(f"  Volatility:   {inv.get('annualized_volatility', np.nan):.2f}%")
            print(f"  Max Drawdown: {inv.get('maximum_drawdown', np.nan):.2f}%")
            print(f"  Calmar:       {inv.get('calmar_ratio', np.nan):.4f}")
            print(f"  Win Rate:     {inv.get('win_rate', np.nan):.2%}")


def save_results(results: dict, args) -> None:
    """
    Save results to files.
    
    Args:
        results: Backtest results dictionary
        args: Command line arguments
    """
    logger.info("Saving results...")
    
    results_dir = get_results_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save summary
    summary = summarize_results(results)
    summary_path = results_dir / f'summary_{timestamp}.csv'
    summary.to_csv(summary_path)
    logger.info(f"Summary saved to {summary_path}")
    
    # Save predictions for each model
    predictions_dir = results_dir / 'predictions' / timestamp
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_results in results.items():
        predictions = model_results['predictions']
        actuals = model_results['actuals']
        
        # Save predictions
        pred_path = predictions_dir / f'{model_name}_predictions.csv'
        predictions.to_csv(pred_path)
        
        # Save actuals
        actual_path = predictions_dir / f'{model_name}_actuals.csv'
        actuals.to_csv(actual_path)
    
    logger.info(f"Predictions saved to {predictions_dir}")
    
    # Save configuration
    config = {
        'start_date': args.start_date,
        'end_date': args.end_date,
        'min_train_size': args.min_train,
        'lags': args.lags,
        'models': args.models,
        'seed': args.seed,
        'timestamp': timestamp,
    }
    
    config_path = results_dir / f'config_{timestamp}.json'
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4, default=str)
    logger.info(f"Config saved to {config_path}")


def main():
    """
    Main orchestrator function.
    """
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file='logs/pipeline.log',
        console=not args.no_console_log
    )
    
    # Set random seed
    set_seed(args.seed)
    
    # Log startup information
    logger.info("=" * 60)
    logger.info("FACTOR TIMING PROJECT")
    logger.info("=" * 60)
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date or 'Today'}")
    logger.info(f"Min training size: {args.min_train} months")
    logger.info(f"Lags: {args.lags}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Seed: {args.seed}")
    
    # Check if any action is specified
    if not any([args.run_all, args.download_data, args.backtest]):
        logger.warning("No action specified. Use --run-all, --download-data, or --backtest.")
        logger.info("Running default: --run-all")
        args.run_all = True
    
    # Initialize variables
    returns_df = None
    prices_df = None
    results = None
    
    # Execute actions
    try:
        # Step 1: Data Pipeline
        if args.run_all or args.download_data:
            returns_df, prices_df = run_data_pipeline(args)
        else:
            # Try to load existing data
            logger.info("Loading existing processed data...")
            returns_df, prices_df = load_processed_data()
            
            if returns_df is None:
                logger.error("No existing data found. Run with --download-data first.")
                sys.exit(1)
        
        # Step 2: Backtest
        if args.run_all or args.backtest:
            if returns_df is not None:
                results = run_backtest_pipeline(args, returns_df)
            else:
                logger.error("No data available for backtest.")
                sys.exit(1)
        
        # Step 3: Display results
        if results:
            display_results(results)
            save_results(results, args)
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()