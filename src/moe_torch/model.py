"""
PyTorch Mixture of Experts model with LSTM gating and MLP experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Expert(nn.Module):
    """
    Expert network - MLP that predicts factor returns for a specific regime.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        hidden_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize expert network.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            output_size: Number of output factors
            hidden_layers: Number of hidden layers (0 = linear)
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.
        
        Args:
            x: Input tensor (batch_size, input_size)
            
        Returns:
            Predictions (batch_size, output_size)
        """
        return self.network(x)


class LSTMGatingNetwork(nn.Module):
    """
    LSTM-based gating network that produces softmax weights over experts.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        lstm_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize LSTM gating network.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            n_experts: Number of experts/regimes
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.gating_head = nn.Linear(hidden_size, n_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through gating network.
        
        Args:
            x: Input sequence (batch_size, seq_length, input_size)
            
        Returns:
            Gating probabilities (batch_size, n_experts)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_size)
        
        # Use last time step's output
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Gating logits
        logits = self.gating_head(last_out)  # (batch_size, n_experts)
        
        # Softmax
        return F.softmax(logits, dim=1)


class TorchMoE(nn.Module):
    """
    Mixture of Experts with LSTM gating and MLP experts.
    
    Architecture:
        Input sequence -> LSTM Gating Network -> Softmax weights
        Input (last step) -> Expert 1, Expert 2, ..., Expert K -> predictions
        Output = sum(weights * predictions)
    """
    
    def __init__(
        self,
        n_experts: int = 4,
        input_size: int = 28,
        output_size: int = 6,
        hidden_size: int = 32,
        lstm_layers: int = 1,
        expert_hidden_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize TorchMoE model.
        
        Args:
            n_experts: Number of experts/regimes
            input_size: Number of input features
            output_size: Number of output factors
            hidden_size: Hidden dimension for LSTM and experts
            lstm_layers: Number of LSTM layers
            expert_hidden_layers: Number of hidden layers in each expert
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_experts = n_experts
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # Gating network (LSTM)
        self.gating = LSTMGatingNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=n_experts,
            lstm_layers=lstm_layers,
            dropout=dropout
        )
        
        # Expert networks (MLPs)
        self.experts = nn.ModuleList([
            Expert(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                hidden_layers=expert_hidden_layers,
                dropout=dropout
            )
            for _ in range(n_experts)
        ])
        
        logger.info(
            f"TorchMoE initialized: {n_experts} experts, "
            f"input_size={input_size}, output_size={output_size}, "
            f"hidden_size={hidden_size}"
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE.
        
        Args:
            x: Input sequence (batch_size, seq_length, input_size)
            
        Returns:
            mixture_pred: Weighted combination (batch_size, output_size)
            gating_probs: Gating probabilities (batch_size, n_experts)
        """
        # Get gating probabilities from sequence
        gating_probs = self.gating(x)  # (batch_size, n_experts)
        
        # Last time step features for experts
        x_last = x[:, -1, :]  # (batch_size, input_size)
        
        # Get predictions from each expert
        expert_preds = torch.stack(
            [expert(x_last) for expert in self.experts],
            dim=1
        )  # (batch_size, n_experts, output_size)
        
        # Weighted combination
        # gating_probs: (batch_size, n_experts) -> unsqueeze for broadcasting
        mixture_pred = torch.sum(
            gating_probs.unsqueeze(-1) * expert_preds,
            dim=1
        )  # (batch_size, output_size)
        
        return mixture_pred, gating_probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions (for evaluation).
        
        Args:
            x: Input sequence (batch_size, seq_length, input_size)
            
        Returns:
            Predictions (batch_size, output_size)
        """
        self.eval()
        with torch.no_grad():
            pred, _ = self.forward(x)
        return pred
    
    def get_regime_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get regime probabilities for interpretability.
        
        Args:
            x: Input sequence (batch_size, seq_length, input_size)
            
        Returns:
            Gating probabilities (batch_size, n_experts)
        """
        self.eval()
        with torch.no_grad():
            _, probs = self.forward(x)
        return probs


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)