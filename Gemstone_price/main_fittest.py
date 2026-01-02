"""
Main script for training gemstone price prediction model
Can be run from notebook using: %run main_fittest.py --config config/Gemstone.yaml
"""
from cli import GemstoneLightningCLI
from module import GemstoneLightningModule
from model import GemstonePriceModel


def cli_main():
    """Main function to run training and testing."""
    cli = GemstoneLightningCLI(run=False)
    
    # Setup datamodule to get input_dim
    cli.datamodule.setup()
    
    # Update model if input_dim is 0 (needs to be set from datamodule)
    # Check the first layer of the model to see if input_dim needs updating
    needs_update = False
    if hasattr(cli.model, 'model') and hasattr(cli.model.model, 'model'):
        # Get the first Linear layer
        for module in cli.model.model.model.modules():
            if hasattr(module, 'in_features'):
                if module.in_features == 0:
                    needs_update = True
                break
    
    if needs_update:
        # Recreate model with correct input_dim
        model_config = cli.config.get("model", {}).get("init_args", {}).get("model", {}).get("init_args", {})
        new_model = GemstonePriceModel(
            input_dim=cli.datamodule.input_dim,
            hidden_layers=model_config.get("hidden_layers", [128, 64, 32]),
            dropout_rates=model_config.get("dropout_rates", [0.15, 0.1, 0.05]),
            activation=model_config.get("activation", "relu"),
        )
        # Create new Lightning module with updated model
        module_config = cli.config.get("model", {}).get("init_args", {})
        cli.model = GemstoneLightningModule(
            model=new_model,
            lr=module_config.get("lr", 0.0001),
            weight_decay=module_config.get("weight_decay", 0.0),
            lr_scheduler_factor=module_config.get("lr_scheduler_factor", 0.5),
            lr_scheduler_patience=module_config.get("lr_scheduler_patience", 5),
            save_dir=module_config.get("save_dir", "models"),
            name=module_config.get("name", "gemstone_price_model"),
        )
    
    # Get checkpoint path from config or CLI argument
    fit_ckpt_path = cli.config.get("fit", {}).get("ckpt_path")
    if fit_ckpt_path is None or fit_ckpt_path == "null":
        fit_ckpt_path = None
    
    # Run training
    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
        ckpt_path=fit_ckpt_path,
    )
    
    # Get test checkpoint path
    test_ckpt_path = cli.config.get("test", {}).get("ckpt_path", "best")
    
    # For testing, use the current trained model (which has test_step method)
    # The model weights are already updated from training
    # If you need to test with a specific checkpoint, load it explicitly in the notebook
    print("Running test with the trained model...")
    cli.trainer.test(
        model=cli.model,
        datamodule=cli.datamodule,
    )


if __name__ == "__main__":
    cli_main()

