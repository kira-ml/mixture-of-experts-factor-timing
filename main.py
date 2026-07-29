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
from typing import Dict

# Import project modules
from src.data_pipeline import DataPipeline, load_processed_data
from src.backtest import backtest_models, summarize_results, get_best_model
from src.evaluation import format_metrics
from src.utils import setup_logging, set_seed, get_data_dir, get_results_dir, load_default_config
from src.visualization import analyze_moe_regimes

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
        default=['persistence', 'rolling_avg', 'momentum', 'linear', 'rf', 'moe'],  # Add 'moe'
        help='Models to run (persistence, rolling_avg, momentum, linear, rf, moe)'
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
        'moe': {'n_experts': 4, 'n_iterations': 100, 'learning_rate': 0.01},  # Add MoE
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
    
    # Load FRED data (if available)
    macro_df = None
    fred_df = None

    try:
        from src.fred_data import load_fred_data
        fred_df = load_fred_data(
            start_date=args.start_date,
            end_date=args.end_date,
            use_cache=True,
            add_transforms=True
        )
        
        # Check if FRED data has enough history for the backtest
        # We need at least min_train_size months of data
        if len(fred_df) >= args.min_train:
            logger.info(f"Loaded FRED data: {fred_df.shape[1]} series, {len(fred_df)} months")
            macro_df = fred_df
        else:
            logger.warning(f"FRED data only has {len(fred_df)} months (< {args.min_train} required). Falling back to VIX.")
            if 'VIX' in returns_df.columns:
                macro_df = returns_df[['VIX']]
                logger.info("Using VIX as macro indicator")
    except Exception as e:
        logger.warning(f"FRED data not available: {e}. Falling back to VIX.")
        if 'VIX' in returns_df.columns:
            macro_df = returns_df[['VIX']]
            logger.info("Using VIX as macro indicator")
    
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


def display_results(results):
    """Display backtest results in a formatted table."""
    summary = summarize_results(results)
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(summary.round(4))
    
    print("\n" + "-" * 80)
    print("BEST PERFORMING MODELS")
    print("-" * 80)
    
    # Helper to safely extract scalar values
    def _fmt(val, decimals=4):
        """Convert numpy array/list to float and format."""
        if isinstance(val, (np.ndarray, list)):
            if len(val) > 0:
                val = float(val[0])
            else:
                val = np.nan
        return f"{val:.{decimals}f}" if not np.isnan(val) else "NaN"
    
    # Best Sharpe
    best_sharpe_model, best_sharpe = get_best_model(results, 'sharpe')
    print(f"Best Sharpe Ratio:    {best_sharpe_model or 'None'} ({_fmt(best_sharpe)})")
    
    # Best Calmar
    best_calmar_model, best_calmar = get_best_model(results, 'calmar')
    print(f"Best Calmar Ratio:    {best_calmar_model or 'None'} ({_fmt(best_calmar)})")
    
    # Best RMSE
    best_rmse_model, best_rmse = get_best_model(results, 'rmse')
    print(f"Lowest RMSE:          {best_rmse_model or 'None'} ({_fmt(best_rmse)})")
    
    # Best Return
    best_return_model, best_return = get_best_model(results, 'ann_return')
    print(f"Best Annual Return:   {best_return_model or 'None'} ({_fmt(best_return)}%)")
    
    # Best Win Rate
    best_wr_model, best_wr = get_best_model(results, 'win_rate')
    print(f"Best Win Rate:        {best_wr_model or 'None'} ({_fmt(best_wr)})")
    
    print("\n" + "=" * 80)
    print("DETAILED METRICS BY MODEL")
    print("=" * 80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        metrics = model_results['metrics']
        
        print(f"  RMSE: {_fmt(metrics.get('rmse', np.nan))}")
        print(f"  MAE:  {_fmt(metrics.get('mae', np.nan))}")
        
        # Per-factor RMSE
        if 'per_factor_rmse' in metrics:
            print("  Per-factor RMSE:")
            for factor, rmse in metrics['per_factor_rmse'].items():
                print(f"    {factor}: {_fmt(rmse)}")
        
        # Investment metrics
        inv = metrics.get('investment', {})
        if inv:
            print(f"  Sharpe:       {_fmt(inv.get('sharpe_ratio', np.nan))}")
            print(f"  Return:       {_fmt(inv.get('annualized_return', np.nan))}%")
            print(f"  Volatility:   {_fmt(inv.get('annualized_volatility', np.nan))}%")
            print(f"  Max Drawdown: {_fmt(inv.get('maximum_drawdown', np.nan))}%")
            print(f"  Calmar:       {_fmt(inv.get('calmar_ratio', np.nan))}")
            print(f"  Win Rate:     {_fmt(inv.get('win_rate', np.nan))}")


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
        
        # Save portfolio returns (NEW - with DatetimeIndex fix)
        if 'portfolio_returns' in model_results:
            portfolio_df = model_results['portfolio_returns']
            if isinstance(portfolio_df, pd.DataFrame):
                # --- FIX: Ensure index is a proper DatetimeIndex before saving ---
                if not isinstance(portfolio_df.index, pd.DatetimeIndex):
                    portfolio_df.index = pd.to_datetime(portfolio_df.index)
                portfolio_df.index.freq = 'ME'  # Force monthly frequency
                portfolio_path = predictions_dir / f'{model_name}_portfolio_returns.csv'
                portfolio_df.to_csv(portfolio_path)
            else:
                # Fallback for backward compatibility (if it's still a numpy array)
                portfolio_path = predictions_dir / f'{model_name}_portfolio_returns.csv'
                pd.DataFrame(
                    model_results['portfolio_returns'],
                    index=model_results['predictions'].index,
                    columns=['portfolio_returns']
                ).to_csv(portfolio_path)
    
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
        
        # Step 4: Regime analysis for MoE (if available)
        if results and 'moe' in results:
            logger.info("=" * 60)
            logger.info("STEP 4: Regime Analysis for MoE")
            logger.info("=" * 60)
            
            fitted_model = results['moe'].get('fitted_model')
            
            if fitted_model is not None:
                try:
                    from src.backtest import prepare_backtest_data
                    from src.fred_data import load_fred_data
                    
                    # Use the SAME macro data that was used in the backtest
                    macro_df = None
                    
                    # Try FRED first (matches backtest behavior)
                    try:
                        fred_df = load_fred_data(
                            start_date=args.start_date,
                            end_date=args.end_date,
                            use_cache=True,
                            add_transforms=True
                        )
                        if len(fred_df) >= args.min_train:
                            macro_df = fred_df
                            logger.info(f"Using FRED data for regime analysis ({macro_df.shape[1]} features)")
                    except Exception as e:
                        logger.warning(f"FRED not available for regime analysis: {e}")
                    
                    # Fallback to VIX if FRED not available
                    if macro_df is None and 'VIX' in returns_df.columns:
                        macro_df = returns_df[['VIX']]
                        logger.info("Using VIX for regime analysis (FRED not available)")
                    
                    # Prepare data with the SAME macro_df used in backtest
                    X, y, dates = prepare_backtest_data(
                        returns_df=returns_df,
                        macro_df=macro_df,
                        min_train_size=args.min_train,
                        test_size=1,
                        lags=args.lags
                    )
                    
                    save_dir = get_results_dir() / 'regime_analysis'
                    regime_df, regime_stats = analyze_moe_regimes(
                        model=fitted_model,
                        X=X,
                        returns_df=returns_df,
                        dates=pd.DatetimeIndex(dates),
                        save_dir=save_dir
                    )
                    
                    logger.info(f"Regime analysis complete. Results saved to {save_dir}")
                    
                except Exception as e:
                    logger.error(f"Regime analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.warning("No fitted MoE model found in results.")

        
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