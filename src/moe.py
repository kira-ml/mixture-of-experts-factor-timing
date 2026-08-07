"""
Simple Mixture of Experts for regime-switching factor timing.
Week 2-3: Minimal implementation for comparison with baselines.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMoE:
    """
    Simple Mixture of Experts model with linear experts and softmax gating.
    
    This is a minimal implementation that is easy to understand and compare
    with baseline models. It uses:
    - Linear regression as experts (one per regime)
    - Softmax gating based on input features
    - Iterative gradient-based estimation (approximating EM)
    
    Architecture:
        Input -> Gating Network (linear) -> softmax -> weights
        Input -> Expert 1 -> prediction_1
        Input -> Expert 2 -> prediction_2
        ...
        Output = sum(weights * predictions)
    """
    
    def __init__(
        self,
        n_experts: int = 3,
        n_iterations: int = 50,
        learning_rate: float = 0.01,
        random_state: int = 42
    ):
        """
        Initialize the Mixture of Experts model.
        
        Args:
            n_experts: Number of experts/regimes (default: 3)
            n_iterations: Number of training iterations (default: 50)
            learning_rate: Learning rate for updates (default: 0.01)
            random_state: Random seed for reproducibility
        """
        self.n_experts = n_experts
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Model components
        self.experts = []  # List of LinearRegression objects
        self.gating_weights = None  # Linear gating network weights
        self.gating_bias = None
        
        # Feature scaling
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Training state
        self.is_fitted = False
        self.n_outputs = None
        self.n_features = None
        self.responsibilities = None  # P(expert | input)
        self.loss_history = []
        
        # Set random seed
        np.random.seed(random_state)
    
    def _initialize_parameters(self, n_features: int, n_outputs: int):
        """
        Initialize model parameters.
        
        Args:
            n_features: Number of input features
            n_outputs: Number of output factors
        """
        self.n_features = n_features
        self.n_outputs = n_outputs
        
        # Initialize gating network weights
        # Shape: (n_features, n_experts) -> output: (n_samples, n_experts)
        self.gating_weights = np.random.randn(n_features, self.n_experts) * 0.1
        self.gating_bias = np.zeros(self.n_experts)
        
        # Initialize experts (one per regime)
        self.experts = []
        for i in range(self.n_experts):
            expert = LinearRegression()
            # Don't fit yet - we'll do it in EM
            self.experts.append(expert)
        
        logger.info(f"Initialized MoE with {self.n_experts} experts, {n_features} features, {n_outputs} outputs")
    
    def _compute_gating(self, X: np.ndarray) -> np.ndarray:
        """
        Compute gating probabilities (softmax over experts).
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Gating probabilities (n_samples, n_experts)
        """
        # Linear transformation
        logits = X @ self.gating_weights + self.gating_bias
        # Softmax
        probs = softmax(logits, axis=1)
        return probs
    
    def _compute_expert_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions from all experts.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Expert predictions (n_samples, n_experts, n_outputs)
        """
        predictions = np.zeros((X.shape[0], self.n_experts, self.n_outputs))
        
        for i, expert in enumerate(self.experts):
            # Check if expert has been fitted (has coef_ attribute)
            if hasattr(expert, 'coef_') and expert.coef_ is not None:
                predictions[:, i, :] = expert.predict(X)
            else:
                # Expert not trained yet, return zeros
                predictions[:, i, :] = 0
        
        return predictions
    
    def _e_step(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Expectation step: compute responsibilities P(expert | input).
        
        Args:
            X: Input features (n_samples, n_features)
            y: Targets (n_samples, n_outputs)
            
        Returns:
            Responsibilities (n_samples, n_experts)
        """
        # Get gating probabilities
        gating_probs = self._compute_gating(X)
        
        # Get expert predictions
        expert_preds = self._compute_expert_predictions(X)
        
        # Compute likelihood of each data point under each expert
        # Using negative squared error as likelihood (Gaussian assumption)
        likelihood = np.zeros((X.shape[0], self.n_experts))
        
        for i in range(self.n_experts):
            # Squared error
            errors = y - expert_preds[:, i, :]
            # Negative squared error (higher = more likely)
            # Scale by number of outputs to make it comparable
            likelihood[:, i] = -np.mean(errors ** 2, axis=1)
        
        # Compute responsibilities (posterior)
        # P(expert | data) ∝ P(data | expert) * P(expert)
        responsibilities = gating_probs * np.exp(likelihood - np.max(likelihood, axis=1, keepdims=True))
        responsibilities = responsibilities / (responsibilities.sum(axis=1, keepdims=True) + 1e-10)
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, y: np.ndarray, responsibilities: np.ndarray):
        """
        Maximization step: update expert parameters and gating network.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Targets (n_samples, n_outputs)
            responsibilities: Responsibilities (n_samples, n_experts)
        """
        n_samples = X.shape[0]
        
        # Update each expert with weighted samples
        for i in range(self.n_experts):
            # Get weights for this expert
            weights = responsibilities[:, i]
            
            # Skip if all weights are zero
            if np.sum(weights) < 1e-10:
                continue
            
            # Weighted least squares for this expert
            # Use Ridge regression with L2 regularization to prevent overfitting
            from sklearn.linear_model import Ridge
            self.experts[i] = Ridge(alpha=0.1)
            self.experts[i].fit(X, y, sample_weight=weights)
        
        # Update gating network using gradient ascent
        # This is a simplified update using the responsibilities as targets
        # For linear gating with softmax, we can use cross-entropy loss
        
        # Compute gating logits
        logits = X @ self.gating_weights + self.gating_bias
        
        # Compute softmax probabilities
        probs = softmax(logits, axis=1)
        
        # Gradient for gating weights: (responsibilities - probs) * X
        grad_weights = X.T @ (responsibilities - probs) / n_samples
        grad_bias = np.mean(responsibilities - probs, axis=0)
        
        # Update with learning rate
        self.gating_weights += self.learning_rate * grad_weights
        self.gating_bias += self.learning_rate * grad_bias
    
    def fit(self, X, y):
        """
        Train the Mixture of Experts model using iterative gradient-based estimation.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target matrix (n_samples, n_outputs)
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        elif isinstance(y, pd.Series):
            y = y.values.reshape(-1, 1)
        
        X = np.array(X)
        y = np.array(y)
        
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        
        # Standardize features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Initialize parameters
        self._initialize_parameters(n_features, n_outputs)
        
        # EM iterations
        logger.info(f"Training MoE with {self.n_iterations} iterations...")
        
        self.loss_history = []
        
        for iteration in range(self.n_iterations):
            # E-step: compute responsibilities
            responsibilities = self._e_step(X_scaled, y_scaled)
            self.responsibilities = responsibilities
            
            # M-step: update parameters
            self._m_step(X_scaled, y_scaled, responsibilities)
            
            # Compute loss (negative log-likelihood)
            gating_probs = self._compute_gating(X_scaled)
            expert_preds = self._compute_expert_predictions(X_scaled)
            
            # Compute mixture prediction
            mixture_pred = np.sum(
                gating_probs[:, :, np.newaxis] * expert_preds,
                axis=1
            )
            
            # Negative log-likelihood (simplified)
            mse = np.mean((y_scaled - mixture_pred) ** 2)
            self.loss_history.append(mse)
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: MSE = {mse:.4f}")
        
        # Ensure model is marked as fitted even if predict fails
        self.is_fitted = True
        
        # Final predictions for a baseline
        try:
            final_pred = self.predict(X)
            final_mse = np.mean((y - final_pred) ** 2)
            logger.info(f"Training complete. Final MSE: {final_mse:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute final MSE: {e}")
            logger.info("Training complete, but final evaluation skipped.")
    
    def predict(self, X):
        """
        Generate predictions from the MoE model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples, n_outputs)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Standardize features
        X_scaled = self.scaler_X.transform(X)
        
        # Compute gating probabilities
        gating_probs = self._compute_gating(X_scaled)
        
        # Compute expert predictions
        expert_preds = self._compute_expert_predictions(X_scaled)
        
        # Mixture prediction (weighted average)
        mixture_pred_scaled = np.sum(
            gating_probs[:, :, np.newaxis] * expert_preds,
            axis=1
        )
        
        # Unscale predictions
        predictions = self.scaler_y.inverse_transform(mixture_pred_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get regime probabilities (gating probabilities).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Regime probabilities (n_samples, n_experts)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_scaled = self.scaler_X.transform(X)
        return self._compute_gating(X_scaled)
    
    def get_regime_probabilities(self, X):
        """Alias for predict_proba."""
        return self.predict_proba(X)
    
    def get_expert_predictions(self, X):
        """
        Get predictions from each expert separately.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Expert predictions (n_samples, n_experts, n_outputs)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_scaled = self.scaler_X.transform(X)
        expert_preds_scaled = self._compute_expert_predictions(X_scaled)
        
        # Unscale each expert's predictions
        expert_preds = np.zeros_like(expert_preds_scaled)
        for i in range(self.n_experts):
            expert_preds[:, i, :] = self.scaler_y.inverse_transform(
                expert_preds_scaled[:, i, :]
            )
        
        return expert_preds


def create_moe(
    n_experts: int = 3,
    n_iterations: int = 50,
    learning_rate: float = 0.01,
    random_state: int = 42
) -> SimpleMoE:
    """
    Factory function to create a MoE model.
    
    Args:
        n_experts: Number of experts/regimes
        n_iterations: Number of EM iterations
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        SimpleMoE instance
    """
    return SimpleMoE(
        n_experts=n_experts,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        random_state=random_state
    )


if __name__ == "__main__":
    # Example usage
    print("Testing Simple MoE...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    n_outputs = 6
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_outputs)
    
    # Test MoE
    moe = SimpleMoE(n_experts=3, n_iterations=20, learning_rate=0.01)
    moe.fit(X, y)
    
    # Make predictions
    pred = moe.predict(X)
    print(f"Predictions shape: {pred.shape}")
    
    # Get regime probabilities
    probs = moe.get_regime_probabilities(X)
    print(f"Regime probabilities shape: {probs.shape}")
    print(f"Average regime probabilities: {np.mean(probs, axis=0)}")
    
    print("\nMoE test complete!")