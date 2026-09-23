"""
Baseline models for factor return prediction.
Week 1 MVP: Simple models with consistent API (fit, predict).
Supports multi-output prediction for multiple factors.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from src.moe import SimpleMoE, create_moe

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PersistenceModel:
    """
    Persistence forecast: next month's return = current month's return.
    This is the simplest baseline with no training.
    Supports multi-output (multiple factors).
    """
    
    def __init__(self):
        self.last_returns = None
        self.is_fitted = False
        self.n_outputs = None
    
    def fit(self, X, y):
        """
        Store the last available returns for prediction.
        
        Args:
            X: Features (not used, kept for API consistency)
            y: Target returns (used to get last values)
        """
        if len(y) > 0:
            # Handle both 1D and 2D y
            if isinstance(y, pd.Series):
                self.last_returns = y.iloc[-1].values if hasattr(y.iloc[-1], 'values') else y.iloc[-1]
                self.n_outputs = 1 if not hasattr(self.last_returns, '__len__') else len(self.last_returns)
            elif isinstance(y, pd.DataFrame):
                self.last_returns = y.iloc[-1].values
                self.n_outputs = len(self.last_returns)
            else:
                # numpy array
                y_array = np.array(y)
                if y_array.ndim == 1:
                    self.last_returns = y_array[-1]
                    self.n_outputs = 1
                else:
                    self.last_returns = y_array[-1]
                    self.n_outputs = y_array.shape[1]
            
            self.is_fitted = True
            logger.info(f"Persistence model fitted (n_outputs={self.n_outputs})")
        else:
            logger.warning("No data provided to fit Persistence model")
    
    def predict(self, X):
        """
        Predict next month's return = last observed return.
        
        Args:
            X: Features (not used, kept for API consistency)
            
        Returns:
            numpy array: Predictions (n_samples x n_outputs)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples = len(X)
        
        if self.n_outputs == 1:
            return np.full(n_samples, self.last_returns)
        else:
            return np.tile(self.last_returns, (n_samples, 1))
    
    def predict_proba(self, X):
        """Not applicable for this model."""
        raise NotImplementedError("Persistence model does not provide probabilistic predictions")


class RollingAverageModel:
    """
    Rolling average forecast: next month's return = average of last N months.
    Supports multi-output (multiple factors).
    """
    
    def __init__(self, window=12):
        """
        Initialize rolling average model.
        
        Args:
            window: Number of months to average (default: 12)
        """
        self.window = window
        self.history = None
        self.is_fitted = False
        self.n_outputs = None
    
    def fit(self, X, y):
        """
        Store historical returns for rolling average calculation.
        
        Args:
            X: Features (not used, kept for API consistency)
            y: Target returns (historical factor returns)
        """
        if len(y) > 0:
            # Handle different input types
            if isinstance(y, pd.DataFrame):
                self.history = y.iloc[-self.window:].values
                self.n_outputs = y.shape[1]
            elif isinstance(y, pd.Series):
                self.history = y.iloc[-self.window:].values.reshape(-1, 1)
                self.n_outputs = 1
            else:
                y_array = np.array(y)
                if y_array.ndim == 1:
                    self.history = y_array[-self.window:].reshape(-1, 1)
                    self.n_outputs = 1
                else:
                    self.history = y_array[-self.window:]
                    self.n_outputs = y_array.shape[1]
            
            self.is_fitted = True
            logger.info(f"Rolling average model fitted (window={self.window}, n_outputs={self.n_outputs})")
        else:
            logger.warning("No data provided to fit RollingAverage model")
    
    def predict(self, X):
        """
        Predict next month's return = average of last N months.
        
        Args:
            X: Features (not used, kept for API consistency)
            
        Returns:
            numpy array: Predictions (n_samples x n_outputs)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if len(self.history) == 0:
            n_samples = len(X)
            return np.zeros((n_samples, self.n_outputs))
        
        avg_returns = np.mean(self.history, axis=0)
        n_samples = len(X)
        
        return np.tile(avg_returns, (n_samples, 1))
    
    def predict_proba(self, X):
        """Not applicable for this model."""
        raise NotImplementedError("Rolling average model does not provide probabilistic predictions")


class MomentumModel:
    """
    Momentum forecast: weighted average of historical returns.
    More recent months get higher weight.
    Supports multi-output (multiple factors).
    """
    
    def __init__(self, window=12, decay=0.9):
        """
        Initialize momentum model with exponential weighting.
        
        Args:
            window: Number of months to include (default: 12)
            decay: Exponential decay factor (0-1), higher = more weight on recent
        """
        self.window = window
        self.decay = decay
        self.history = None
        self.is_fitted = False
        self.n_outputs = None
    
    def fit(self, X, y):
        """
        Store historical returns for momentum calculation.
        
        Args:
            X: Features (not used, kept for API consistency)
            y: Target returns (historical factor returns)
        """
        if len(y) > 0:
            # Handle different input types
            if isinstance(y, pd.DataFrame):
                self.history = y.iloc[-self.window:].values
                self.n_outputs = y.shape[1]
            elif isinstance(y, pd.Series):
                self.history = y.iloc[-self.window:].values.reshape(-1, 1)
                self.n_outputs = 1
            else:
                y_array = np.array(y)
                if y_array.ndim == 1:
                    self.history = y_array[-self.window:].reshape(-1, 1)
                    self.n_outputs = 1
                else:
                    self.history = y_array[-self.window:]
                    self.n_outputs = y_array.shape[1]
            
            self.is_fitted = True
            logger.info(f"Momentum model fitted (window={self.window}, decay={self.decay}, n_outputs={self.n_outputs})")
        else:
            logger.warning("No data provided to fit Momentum model")
    
    def _calculate_weights(self, n):
        """
        Calculate exponentially decaying weights.
        Most recent observation gets highest weight.
        
        Args:
            n: Number of periods
            
        Returns:
            numpy array: Weights summing to 1
        """
        # Exponential weights: most recent gets highest weight
        # i=0 -> most recent (highest weight), i=n-1 -> oldest (lowest weight)
        weights = np.array([self.decay ** i for i in range(n)])
        weights = weights / weights.sum()  # Normalize to sum to 1
        return weights
    
    def predict(self, X):
        """
        Predict next month's return = weighted average of historical returns.
        
        Args:
            X: Features (not used, kept for API consistency)
            
        Returns:
            numpy array: Predictions (n_samples x n_outputs)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if len(self.history) == 0:
            n_samples = len(X)
            return np.zeros((n_samples, self.n_outputs))
        
        # Calculate weighted average for each output
        weights = self._calculate_weights(len(self.history))
        weighted_avg = np.sum(self.history * weights[:, np.newaxis], axis=0)
        
        n_samples = len(X)
        return np.tile(weighted_avg, (n_samples, 1))
    
    def predict_proba(self, X):
        """Not applicable for this model."""
        raise NotImplementedError("Momentum model does not provide probabilistic predictions")


class LinearRegressionModel:
    """
    Linear regression for factor return prediction.
    Uses sklearn's LinearRegression with optional standardization.
    Supports multi-output (multiple factors).
    """
    
    def __init__(self, standardize=True):
        """
        Initialize linear regression model.
        
        Args:
            standardize: Whether to standardize features (default: True)
        """
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.model = LinearRegression()
        self.is_fitted = False
        self.feature_names = None
        self.n_outputs = None
    
    def fit(self, X, y):
        """
        Train linear regression model.
        
        Args:
            X: Feature matrix (lagged returns + macro indicators)
            y: Target returns (next month's factor returns)
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = np.array(X)
        
        if isinstance(y, pd.DataFrame):
            y_array = y.values
            self.n_outputs = y.shape[1]
        elif isinstance(y, pd.Series):
            y_array = y.values.reshape(-1, 1)
            self.n_outputs = 1
        else:
            y_array = np.array(y)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
                self.n_outputs = 1
            else:
                self.n_outputs = y_array.shape[1]
        
        # Standardize features
        if self.standardize:
            X_array = self.scaler.fit_transform(X_array)
            logger.info("Features standardized")
        
        # Fit model
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        
        logger.info(f"Linear regression fitted (n_features={X_array.shape[1]}, n_outputs={self.n_outputs})")
        
        # Log feature importance if available
        if hasattr(self.model, 'coef_') and self.feature_names:
            # For multi-output, take mean absolute coefficient across outputs
            coefs = self.model.coef_
            if coefs.ndim == 1:
                coef_flat = coefs
            else:
                coef_flat = np.mean(np.abs(coefs), axis=0)
            coef_dict = dict(zip(self.feature_names, coef_flat))
            top_features = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            logger.info(f"Top 3 features: {top_features}")
    
    def predict(self, X):
        """
        Generate predictions from linear regression.
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy array: Predictions (n_samples x n_outputs)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # Standardize features
        if self.standardize:
            X_array = self.scaler.transform(X_array)
        
        predictions = self.model.predict(X_array)
        
        # Ensure 2D output
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
    
    def predict_proba(self, X):
        """Not applicable for this model."""
        raise NotImplementedError("Linear regression does not provide probabilistic predictions")
    
    def get_coefficients(self):
        """
        Get model coefficients for interpretation.
        
        Returns:
            dict: Feature coefficients if available
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.feature_names:
            coefs = self.model.coef_
            if coefs.ndim == 1:
                return dict(zip(self.feature_names, coefs))
            else:
                # Return mean coefficients across outputs
                return dict(zip(self.feature_names, np.mean(coefs, axis=0)))
        else:
            return self.model.coef_


class RandomForestModel:
    """
    Random Forest for factor return prediction.
    Uses sklearn's RandomForestRegressor with sensible defaults.
    Supports multi-output (multiple factors).
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=10, random_state=42):
        """
        Initialize random forest model.
        
        Args:
            n_estimators: Number of trees (default: 100)
            max_depth: Maximum tree depth (default: 10)
            min_samples_split: Minimum samples to split a node (default: 10)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,  # Use all cores
        )
        self.is_fitted = False
        self.feature_names = None
        self.feature_importances_ = None
        self.n_outputs = None
    
    def fit(self, X, y):
        """
        Train random forest model.
        
        Args:
            X: Feature matrix (lagged returns + macro indicators)
            y: Target returns (next month's factor returns)
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = np.array(X)
        
        if isinstance(y, pd.DataFrame):
            y_array = y.values
            self.n_outputs = y.shape[1]
        elif isinstance(y, pd.Series):
            y_array = y.values.reshape(-1, 1)
            self.n_outputs = 1
        else:
            y_array = np.array(y)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
                self.n_outputs = 1
            else:
                self.n_outputs = y_array.shape[1]
        
        # Fit model
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        logger.info(f"Random forest fitted (n_estimators={self.model.n_estimators}, n_features={X_array.shape[1]}, n_outputs={self.n_outputs})")
        
        # Log top features
        if self.feature_names and self.feature_importances_ is not None:
            importances = dict(zip(self.feature_names, self.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"Top 3 features: {top_features}")
    
    def predict(self, X):
        """
        Generate predictions from random forest.
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy array: Predictions (n_samples x n_outputs)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        predictions = self.model.predict(X_array)
        
        # Ensure 2D output
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
    
    def predict_proba(self, X):
        """Not applicable for regression model."""
        raise NotImplementedError("Random forest regression does not provide probabilistic predictions")
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            dict: Feature name -> importance score
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.feature_names and self.feature_importances_ is not None:
            return dict(zip(self.feature_names, self.feature_importances_))
        else:
            return self.feature_importances_




# Add this to the model creation function
def create_model(model_name, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name: Name of the model ('persistence', 'rolling_avg', 'momentum', 'linear', 'rf', 'moe')
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    model_map = {
        'persistence': PersistenceModel,
        'rolling_avg': RollingAverageModel,
        'momentum': MomentumModel,
        'linear': LinearRegressionModel,
        'rf': RandomForestModel,
        'moe': SimpleMoE,  # Add MoE
    }
    
    model_name = model_name.lower()
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(model_map.keys())}")
    
    return model_map[model_name](**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Testing models with multi-output...")
    
    # Create dummy data with multiple outputs
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_outputs = 6  # 6 factors
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_outputs)
    
    # Test each model
    models = {
        'persistence': PersistenceModel(),
        'rolling_avg': RollingAverageModel(window=6),
        'momentum': MomentumModel(window=12),
        'linear': LinearRegressionModel(standardize=True),
        'rf': RandomForestModel(n_estimators=10, max_depth=5),
    }
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        try:
            # Fit
            model.fit(X[:-1], y[:-1])
            
            # Predict
            pred = model.predict(X[-1:])
            print(f"  Prediction shape: {pred.shape}")
            print(f"  First prediction: {pred[0][:3]}...")
            
            # Check if fitted
            print(f"  Is fitted: {model.is_fitted}")
            print(f"  Number of outputs: {model.n_outputs if hasattr(model, 'n_outputs') else 'N/A'}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nAll models tested successfully!")