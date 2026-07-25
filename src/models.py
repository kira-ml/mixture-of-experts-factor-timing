"""
Baseline models for factor return prediction.
Week 1 MVP: Simple models with consistent API (fit, predict).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PersistenceModel:
    """
    Persistence forecast: next month's return = current month's return.
    This is the simplest baseline with no training.
    """
    
    def __init__(self):
        self.last_return = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Store the last available return for prediction.
        
        Args:
            X: Features (not used, kept for API consistency)
            y: Target returns (used to get last value)
        """
        if len(y) > 0:
            self.last_return = y.iloc[-1] if isinstance(y, pd.Series) else y[-1]
            self.is_fitted = True
            logger.info("Persistence model fitted (stored last return)")
        else:
            logger.warning("No data provided to fit Persistence model")
    
    def predict(self, X):
        """
        Predict next month's return = last observed return.
        
        Args:
            X: Features (not used, kept for API consistency)
            
        Returns:
            numpy array: Predictions (same shape as number of rows in X)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples = len(X)
        return np.full(n_samples, self.last_return)
    
    def predict_proba(self, X):
        """Not applicable for this model."""
        raise NotImplementedError("Persistence model does not provide probabilistic predictions")


class RollingAverageModel:
    """
    Rolling average forecast: next month's return = average of last N months.
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
    
    def fit(self, X, y):
        """
        Store historical returns for rolling average calculation.
        
        Args:
            X: Features (not used, kept for API consistency)
            y: Target returns (historical factor returns)
        """
        if len(y) > 0:
            self.history = y.iloc[-self.window:] if isinstance(y, pd.Series) else y[-self.window:]
            self.is_fitted = True
            logger.info(f"Rolling average model fitted (window={self.window})")
        else:
            logger.warning("No data provided to fit RollingAverage model")
    
    def predict(self, X):
        """
        Predict next month's return = average of last N months.
        
        Args:
            X: Features (not used, kept for API consistency)
            
        Returns:
            numpy array: Predictions (same shape as number of rows in X)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        avg_return = np.mean(self.history) if len(self.history) > 0 else 0
        
        n_samples = len(X)
        return np.full(n_samples, avg_return)
    
    def predict_proba(self, X):
        """Not applicable for this model."""
        raise NotImplementedError("Rolling average model does not provide probabilistic predictions")


class MomentumModel:
    """
    Momentum forecast: weighted average of historical returns.
    More recent months get higher weight.
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
    
    def fit(self, X, y):
        """
        Store historical returns for momentum calculation.
        
        Args:
            X: Features (not used, kept for API consistency)
            y: Target returns (historical factor returns)
        """
        if len(y) > 0:
            # Take last 'window' months
            self.history = y.iloc[-self.window:] if isinstance(y, pd.Series) else y[-self.window:]
            self.is_fitted = True
            logger.info(f"Momentum model fitted (window={self.window}, decay={self.decay})")
        else:
            logger.warning("No data provided to fit Momentum model")
    
    def _calculate_weights(self, n):
        """
        Calculate exponentially decaying weights.
        
        Args:
            n: Number of periods
            
        Returns:
            numpy array: Weights summing to 1
        """
        # Exponential weights: most recent gets highest weight
        weights = np.array([self.decay ** (n - i) for i in range(1, n + 1)])
        weights = weights / weights.sum()  # Normalize to sum to 1
        return weights
    
    def predict(self, X):
        """
        Predict next month's return = weighted average of historical returns.
        
        Args:
            X: Features (not used, kept for API consistency)
            
        Returns:
            numpy array: Predictions (same shape as number of rows in X)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if len(self.history) == 0:
            return np.full(len(X), 0)
        
        # Calculate weighted average
        weights = self._calculate_weights(len(self.history))
        weighted_avg = np.sum(self.history * weights)
        
        n_samples = len(X)
        return np.full(n_samples, weighted_avg)
    
    def predict_proba(self, X):
        """Not applicable for this model."""
        raise NotImplementedError("Momentum model does not provide probabilistic predictions")


class LinearRegressionModel:
    """
    Linear regression for factor return prediction.
    Uses sklearn's LinearRegression with optional standardization.
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
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Standardize features
        if self.standardize:
            X_array = self.scaler.fit_transform(X_array)
            logger.info("Features standardized")
        
        # Fit model
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        
        logger.info(f"Linear regression fitted (n_features={X_array.shape[1]})")
        
        # Log feature importance if available
        if hasattr(self.model, 'coef_') and self.feature_names:
            coefs = dict(zip(self.feature_names, self.model.coef_))
            logger.info(f"Top 3 features: {sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]}")
    
    def predict(self, X):
        """
        Generate predictions from linear regression.
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy array: Predictions
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
        
        return self.model.predict(X_array)
    
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
            return dict(zip(self.feature_names, self.model.coef_))
        else:
            return self.model.coef_


class RandomForestModel:
    """
    Random Forest for factor return prediction.
    Uses sklearn's RandomForestRegressor with sensible defaults.
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
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Fit model
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        logger.info(f"Random forest fitted (n_estimators={self.model.n_estimators}, n_features={X_array.shape[1]})")
        
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
            numpy array: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        return self.model.predict(X_array)
    
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


def create_model(model_name, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name: Name of the model ('persistence', 'rolling_avg', 'momentum', 'linear', 'rf')
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
    }
    
    model_name = model_name.lower()
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(model_map.keys())}")
    
    return model_map[model_name](**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Testing models...")
    
    # Create dummy data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    # Test each model
    models = {
        'persistence': PersistenceModel(),
        'rolling_avg': RollingAverageModel(window=6),
        'momentum': MomentumModel(window=12),
        'linear': LinearRegressionModel(),
        'rf': RandomForestModel(n_estimators=10, max_depth=5),
    }
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        try:
            # Fit
            model.fit(X[:-1], y[:-1])
            
            # Predict
            pred = model.predict(X[-1:])
            print(f"  Prediction: {pred[0]:.4f}")
            
            # Check if fitted
            print(f"  Is fitted: {model.is_fitted}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nAll models tested successfully!")