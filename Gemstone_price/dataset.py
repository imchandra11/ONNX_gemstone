"""
Gemstone Dataset
PyTorch Dataset class for gemstone price prediction
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class GemstoneDataset(Dataset):
    """PyTorch Dataset for gemstone price prediction."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Preprocessed feature array (numeric + encoded categorical)
            targets: Price targets array
        """
        self.x = torch.from_numpy(features).float()
        self.y = torch.from_numpy(targets).float().view(-1, 1)

    def __len__(self):
        """Return dataset size."""
        return len(self.x)

    def __getitem__(self, idx):
        """Get a single sample."""
        return self.x[idx], self.y[idx]

