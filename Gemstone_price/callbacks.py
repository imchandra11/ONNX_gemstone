"""
Custom callbacks for Gemstone Price Prediction
"""
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint


class ONNXExportCallback(L.Callback):
    """Callback to export model to ONNX format when best checkpoint is saved."""

    def __init__(
        self,
        output_dir: str = "models",
        model_name: str = "gemstone_price_model",
        input_dim: Optional[int] = None,
    ):
        """
        Initialize ONNX export callback.
        
        Args:
            output_dir: Directory to save ONNX model
            model_name: Name for the ONNX model file
            input_dim: Input dimension for dummy input. If None, will try to infer.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.input_dim = input_dim
        self.last_best_path = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Export to ONNX after validation if best model was saved."""
        # Check if best checkpoint was just saved
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if hasattr(callback, "best_model_path") and callback.best_model_path:
                    # Check if this is a new best model
                    if callback.best_model_path != self.last_best_path:
                        self.last_best_path = callback.best_model_path
                        self._export_to_onnx(pl_module, trainer)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Export final model to ONNX at the end of training."""
        # Export the best model one final time
        self._export_to_onnx(pl_module, trainer)

    def _export_to_onnx(self, pl_module: L.LightningModule, trainer: L.Trainer) -> None:
        """Export model to ONNX format."""
        try:
            pl_module.eval()
            
            # Determine input dimension
            input_dim = self.input_dim
            if input_dim is None:
                # Try to get from datamodule
                if hasattr(trainer, "datamodule") and hasattr(trainer.datamodule, "input_dim"):
                    input_dim = trainer.datamodule.input_dim
                else:
                    # Try to infer from model
                    for module in pl_module.model.modules():
                        if hasattr(module, "in_features"):
                            input_dim = module.in_features
                            break
            
            if input_dim is None:
                print("Warning: Could not determine input_dim. Skipping ONNX export.")
                return

            # Create dummy input
            dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)

            # Export to ONNX
            onnx_path = self.output_dir / f"{self.model_name}.onnx"
            torch.onnx.export(
                pl_module.model,  # Export the underlying model, not the Lightning module
                dummy_input,
                onnx_path,
                input_names=["features"],
                output_names=["price"],
                dynamic_axes={"features": {0: "batch"}, "price": {0: "batch"}},
                opset_version=17,
            )

            print(f"Model exported to ONNX: {onnx_path}")
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")

