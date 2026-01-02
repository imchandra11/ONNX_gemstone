"""
Gemstone Lightning Module
Training logic, loss, metrics, optimizer, and scheduler configuration
"""
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import torch
from torch import nn

from model import GemstonePriceModel


class GemstoneLightningModule(L.LightningModule):
    """Lightning Module for gemstone price prediction regression."""

    def __init__(
        self,
        model: GemstonePriceModel,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 5,
        save_dir: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize Lightning module.
        
        Args:
            model: GemstonePriceModel instance
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            lr_scheduler_factor: Factor for ReduceLROnPlateau
            lr_scheduler_patience: Patience for ReduceLROnPlateau
            save_dir: Directory to save ONNX model
            name: Model name for saving
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.MSELoss()
        self.save_dir = save_dir
        self.name = name or "gemstone_price_model"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        mae = torch.mean(torch.abs(preds - y))
        rmse = torch.sqrt(torch.mean((preds - y) ** 2))
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        mae = torch.mean(torch.abs(preds - y))
        rmse = torch.sqrt(torch.mean((preds - y) ** 2))
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        x, y = batch
        preds = self(x)
        return preds

    def export_to_onnx(self, output_path: Optional[Path] = None, input_dim: Optional[int] = None):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model. If None, uses save_dir/name.onnx
            input_dim: Input dimension for dummy input. If None, tries to infer from model.
        """
        self.eval()
        
        if output_path is None:
            if self.save_dir is None:
                raise ValueError("Either output_path or save_dir must be provided")
            output_path = Path(self.save_dir) / f"{self.name}.onnx"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine input dimension
        if input_dim is None:
            # Try to get from model's first layer
            first_layer = None
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    first_layer = module
                    break
            if first_layer is None:
                raise ValueError("Could not determine input_dim. Please provide it explicitly.")
            input_dim = first_layer.in_features
        
        # Create dummy input
        dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["features"],
            output_names=["price"],
            dynamic_axes={"features": {0: "batch"}, "price": {0: "batch"}},
            opset_version=17,
        )
        
        print(f"Model exported to ONNX: {output_path}")
        return output_path

