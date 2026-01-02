"""
Custom Lightning CLI
Extends LightningCLI with argument linking and custom arguments
"""
from lightning.pytorch.cli import LightningCLI


class GemstoneLightningCLI(LightningCLI):
    """Custom CLI with argument linking for gemstone price prediction."""

    def add_arguments_to_parser(self, parser):
        """Add custom arguments and link data to model."""
        # For RESUME training
        parser.add_argument("--fit.ckpt_path", default=None)
        # Select last or best checkpoint for testing
        parser.add_argument("--test.ckpt_path", default="best")
        
        # Note: input_dim is computed after datamodule.setup(), so we can't link it directly
        # The model accepts input_dim=0 initially and will be updated after setup

