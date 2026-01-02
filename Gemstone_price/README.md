# Gemstone Price Prediction Project

A modular PyTorch Lightning-based pipeline for predicting gemstone prices using machine learning.

## Project Structure

```
Gemstone_price/
├── config/
│   └── Gemstone.yaml          # Training configuration
├── dataset.py                  # PyTorch Dataset class
├── datamodule.py              # Lightning DataModule
├── model.py                   # Neural network model
├── module.py                  # Lightning Module
├── cli.py                     # Custom LightningCLI
├── main_fittest.py            # Training script
├── Gemstone_price_trainer.ipynb  # Training notebook
├── gemstone.csv               # Dataset
└── app/                       # FastAPI application
    ├── main.py                # FastAPI routes
    ├── templates/
    │   └── index.html         # Web UI
    └── static/
        └── styles.css         # CSS styling
```

## Prerequisites

1. **Python 3.8+**
2. **Required packages:**
   ```bash
   pip install torch lightning pytorch-lightning
   pip install pandas numpy scikit-learn
   pip install onnx onnxruntime
   pip install fastapi uvicorn jinja2 python-multipart
   pip install tensorboard
   pip install joblib
   ```

## How to Run

### Method 1: Training via Jupyter Notebook (Recommended)

1. **Open the notebook:**
   ```bash
   jupyter notebook Gemstone_price_trainer.ipynb
   ```

2. **Run the cells in order:**
   - Cell 1: Imports the necessary modules
   - Cell 2: Runs training using `%run main_fittest.py --config config/Gemstone.yaml`
   - Subsequent cells: Handle model export, evaluation, and inference

3. **Execute training:**
   ```python
   from module import GemstoneLightningModule
   from datamodule import GemstoneDataModule
   
   %run main_fittest.py --config config/Gemstone.yaml
   ```

### Method 2: Training via Command Line

1. **Navigate to the project directory:**
   ```bash
   cd Gemstone_price
   ```

2. **Run the training script:**
   ```bash
   python main_fittest.py --config config/Gemstone.yaml
   ```

   Or with multiple config files:
   ```bash
   python main_fittest.py --config config/Gemstone.yaml --config config/Gemstone.local.yaml
   ```

### Method 3: Using Lightning CLI Directly

```bash
lightning run model fit --config config/Gemstone.yaml
```

## Configuration

All hyperparameters are configured in `config/Gemstone.yaml`:

- **Model architecture**: `hidden_layers`, `dropout_rates`, `activation`
- **Training**: `max_epochs`, `batch_size`, `learning_rate`
- **Data**: `csv_path`, `val_split`, `categorical_cols`, `numeric_cols`
- **Callbacks**: ModelCheckpoint (best and last), TensorBoard logger

## Training Output

After training, you'll find:

- **Checkpoints**: Saved in `lightning_logs/GemstonePriceTraining/version_X/checkpoints/`
  - `{epoch}-{val_loss:.2f}.best.ckpt` - Best model
  - `{epoch}.last.ckpt` - Last epoch model
- **TensorBoard logs**: `lightning_logs/GemstonePriceTraining/`
- **Preprocessor**: `models/preprocessor.joblib`
- **ONNX model**: Exported via notebook (see Cell 3)

## Running FastAPI for Inference

1. **Ensure you have trained the model** (ONNX export happens automatically during training)

2. **Verify required files exist:**
   - `models/gemstone_price_model.onnx` - ONNX model (auto-exported)
   - `models/preprocessor.joblib` - Preprocessor (auto-saved)

3. **Start the FastAPI server:**

   **Option A: Using the run script (Recommended)**
   ```bash
   cd Gemstone_price
   python run_app.py
   ```

   **Option B: Using uvicorn directly**
   ```bash
   cd Gemstone_price
   uvicorn app.main:app --reload
   ```

   **Option C: Using uvicorn with custom host/port**
   ```bash
   cd Gemstone_price
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the web interface:**
   - Open browser: `http://localhost:8000`
   - Fill in gemstone characteristics:
     - Carat, Depth, Table (numeric)
     - X, Y, Z dimensions (numeric)
     - Cut, Color, Clarity (dropdowns)
   - Click "Predict Price" to get predictions

5. **API endpoints:**
   - `GET /` - Web form interface
   - `POST /predict` - Prediction endpoint
   - `GET /docs` - Interactive API documentation (Swagger UI)
   - `GET /redoc` - Alternative API documentation

## Notebook Workflow

The notebook (`Gemstone_price_trainer.ipynb`) includes:

1. **Training** - Run training via CLI
2. **Model Export** - Export best model to ONNX format
3. **Metadata Saving** - Save preprocessor and model metadata
4. **Evaluation** - Validate model performance
5. **Inference Examples** - Single and batch predictions
6. **Visualization** - Training curves from TensorBoard

## Troubleshooting

### Issue: `input_dim` is 0
- **Solution**: The model automatically handles this. The `main_fittest.py` script sets up the datamodule first, then updates the model with the correct `input_dim`.

### Issue: ModuleNotFoundError
- **Solution**: Make sure you're in the `Gemstone_price` directory and all dependencies are installed.

### Issue: ONNX export fails
- **Solution**: Ensure the model is trained first and the checkpoint path is correct.

### Issue: FastAPI can't find model
- **Solution**: Make sure `models/metadata.json` and `models/gemstone_price_model.onnx` exist after training and ONNX export.

## Example Usage

### Training with custom config:
```bash
python main_fittest.py --config config/Gemstone.yaml --trainer.max_epochs 50
```

### Resume training:
Edit `config/Gemstone.yaml`:
```yaml
fit:
  ckpt_path: "lightning_logs/GemstonePriceTraining/version_0/checkpoints/epoch=10-val_loss=1234.56.best.ckpt"
```

Or via CLI:
```bash
python main_fittest.py --config config/Gemstone.yaml --fit.ckpt_path "path/to/checkpoint.ckpt"
```

## Viewing Training Progress

1. **TensorBoard:**
   ```bash
   tensorboard --logdir lightning_logs/GemstonePriceTraining
   ```
   Then open `http://localhost:6006`

2. **Check logs in notebook** - Training progress is displayed in the notebook output

## Notes

- The dataset (`gemstone.csv`) should be in the `Gemstone_price` directory
- Model checkpoints are saved automatically during training
- The preprocessor is saved automatically after datamodule setup
- **ONNX export happens automatically during training** via the ONNXExportCallback
- The ONNX model will be available in `models/gemstone_price_model.onnx` after training

