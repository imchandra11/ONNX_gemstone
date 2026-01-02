"""
Gemstone Price Model
Configurable MLP architecture for gemstone price prediction
"""
from typing import List, Optional

import torch
from torch import nn


def get_activation(activation: str):
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
    }
    if activation.lower() not in activations:
        raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
    return activations[activation.lower()]()


class GemstonePriceModel(nn.Module):
    """Configurable MLP model for gemstone price prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rates: Optional[List[float]] = None,
        activation: str = "relu",
    ):
        """
        Initialize model.
        
        Args:
            input_dim: Input feature dimension
            hidden_layers: List of hidden layer sizes
            dropout_rates: List of dropout rates (one per hidden layer). If None, no dropout.
            activation: Activation function name (relu, gelu, tanh, sigmoid, leaky_relu)
        """
        super().__init__()
        
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_layers)
        elif len(dropout_rates) != len(hidden_layers):
            raise ValueError(f"dropout_rates length ({len(dropout_rates)}) must match hidden_layers length ({len(hidden_layers)})")
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            if dropout_rates[i] > 0:
                layers.append(nn.Dropout(dropout_rates[i]))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

