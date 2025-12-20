# Gemstone Price Prediction System

A complete machine learning pipeline for predicting gemstone (diamond) prices using neural networks. This project includes model training with PyTorch Lightning, ONNX model export, and a FastAPI web application with an interactive HTML interface.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Model Architecture](#model-architecture)
- [File Descriptions](#file-descriptions)

## 🎯 Overview

This project implements an end-to-end machine learning system for predicting diamond prices based on various attributes such as carat, cut quality, color, clarity, and physical dimensions. The system uses a neural network trained with PyTorch Lightning and serves predictions through a FastAPI web interface.

## ✨ Features

- **Neural Network Model**: Deep learning model using PyTorch Lightning for price prediction
- **ONNX Export**: Model exported to ONNX format for efficient inference
- **Web Interface**: User-friendly HTML form for inputting diamond attributes
- **FastAPI Backend**: High-performance API server for model inference
- **Data Preprocessing**: Automated handling of categorical and numerical features
- **Model Checkpointing**: Automatic saving of best model during training
- **Early Stopping**: Prevents overfitting with early stopping callback

## 📁 Project Structure

```
New folder/
├── gemstone.csv              # Dataset file
├── train.py                  # Training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/                   # Generated after training
│   ├── diamond_regressor.onnx
│   ├── preprocessor.joblib
│   ├── metadata.json
│   └── diamond-regressor.ckpt
└── app/
    ├── main.py              # FastAPI application
    ├── templates/
    │   └── index.html       # Web interface template
    └── static/
        └── styles.css       # CSS styling
```

## 📊 Dataset

The dataset contains information about diamonds with the following features:

### Independent Variables:
- **id**: Unique identifier of each diamond
- **carat**: Weight of the diamond (ct.)
- **cut**: Quality of diamond cut (Fair, Good, Very Good, Premium, Ideal)
- **color**: Color of diamond (D, E, F, G, H, I, J)
- **clarity**: Diamond clarity grade (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
- **depth**: Height of diamond in millimeters
- **table**: Width of the top facet relative to widest point
- **x**: Length in millimeters
- **y**: Width in millimeters
- **z**: Height in millimeters

### Target Variable:
- **price**: Price of the diamond (USD)

**Dataset Source**: [Kaggle Playground Series S3E8](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

## 🛠 Technologies Used

- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: High-level PyTorch wrapper for training
- **ONNX**: Model interoperability format
- **ONNX Runtime**: Efficient inference engine
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Data preprocessing (StandardScaler)
- **Jinja2**: Template engine for HTML rendering

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd "C:\Users\91838\Desktop\New folder"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   
   **Windows (PowerShell)**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   
   **Windows (Command Prompt)**:
   ```cmd
   .venv\Scripts\activate.bat
   ```
   
   **Linux/Mac**:
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Training the Model

Before running the web application, you need to train the model and export it to ONNX format.

### Basic Training

```bash
python train.py --csv gemstone.csv --output-dir models --epochs 30 --batch-size 256 --lr 0.001
```

### Training Arguments

- `--csv`: Path to the CSV dataset file (default: `gemstone.csv`)
- `--output-dir`: Directory to save model artifacts (default: `models`)
- `--epochs`: Number of training epochs (default: `30`)
- `--batch-size`: Batch size for training (default: `256`)
- `--lr`: Learning rate (default: `0.001`)

### Training Process

1. **Data Loading**: Loads and preprocesses the dataset
2. **Feature Engineering**:
   - Categorical features (cut, color, clarity) are one-hot encoded
   - Numerical features (carat, depth, table, x, y, z) are standardized
3. **Train/Validation Split**: 80/20 split with random seed 42
4. **Model Training**: Trains neural network with:
   - Early stopping (patience: 10 epochs)
   - Model checkpointing (saves best model based on validation loss)
   - Learning rate scheduling (reduces LR on plateau)
5. **Model Export**: Exports best model to ONNX format
6. **Artifacts Saved**:
   - `diamond_regressor.onnx`: ONNX model file
   - `preprocessor.joblib`: Preprocessing pipeline
   - `metadata.json`: Model metadata
   - `diamond-regressor.ckpt`: PyTorch Lightning checkpoint

### Example Output

After training, you should see:
```
Training complete. ONNX saved to models/diamond_regressor.onnx
Preprocessor saved to models/preprocessor.joblib
```

## 🌐 Running the Application

Once the model is trained, you can start the FastAPI web server:

```bash
uvicorn app.main:app --reload --port 8000
```

The application will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc (ReDoc)

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:8000`
2. Fill in the diamond attributes:
   - **Carat**: Weight of the diamond (e.g., 1.5)
   - **Cut**: Select from dropdown (Fair, Good, Very Good, Premium, Ideal)
   - **Color**: Select from dropdown (D, E, F, G, H, I, J)
   - **Clarity**: Select from dropdown (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
   - **Depth**: Depth percentage (e.g., 62.5)
   - **Table**: Table percentage (e.g., 58.0)
   - **X, Y, Z**: Dimensions in millimeters
3. Click **"Predict Price"** button
4. View the predicted price displayed on the page

## 🔌 API Endpoints

### GET `/`
Returns the HTML form for price prediction.

**Response**: HTML page with input form

### POST `/predict`
Predicts diamond price based on input features.

**Form Parameters**:
- `carat` (float): Weight of diamond
- `cut` (string): Cut quality
- `color` (string): Color grade
- `clarity` (string): Clarity grade
- `depth` (float): Depth percentage
- `table` (float): Table percentage
- `x` (float): Length in mm
- `y` (float): Width in mm
- `z` (float): Height in mm

**Response**: HTML page with predicted price

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "carat=1.5&cut=Premium&color=G&clarity=VS2&depth=62.5&table=58.0&x=7.5&y=7.6&z=4.7"
```

## 🧠 Model Architecture

The neural network model consists of:

- **Input Layer**: Receives preprocessed features (one-hot encoded categorical + standardized numerical)
- **Hidden Layer 1**: 128 neurons with ReLU activation and 10% dropout
- **Hidden Layer 2**: 64 neurons with ReLU activation and 5% dropout
- **Output Layer**: Single neuron for price prediction

**Training Configuration**:
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (configurable)
- **Scheduler**: ReduceLROnPlateau (reduces LR by 50% when validation loss plateaus)
- **Metrics**: Validation Loss (MSE) and Mean Absolute Error (MAE)

## 📄 File Descriptions

### `train.py`
Main training script that:
- Loads and preprocesses the dataset
- Defines the neural network architecture
- Trains the model using PyTorch Lightning
- Exports the model to ONNX format
- Saves preprocessing pipeline and metadata

### `app/main.py`
FastAPI application that:
- Loads the ONNX model and preprocessor
- Serves the HTML interface
- Handles prediction requests
- Processes input features and returns predictions

### `app/templates/index.html`
HTML template with:
- Input form for diamond attributes
- Dropdown menus for categorical features
- Display area for prediction results
- Modern, responsive design

### `app/static/styles.css`
CSS stylesheet providing:
- Clean, modern UI design
- Responsive layout
- Form styling
- Result display formatting

### `requirements.txt`
Python package dependencies with specific versions for reproducibility.

## 🔧 Troubleshooting

### Issue: Model files not found
**Solution**: Make sure you've run `train.py` first to generate the model files in the `models/` directory.

### Issue: Port already in use
**Solution**: Use a different port:
```bash
uvicorn app.main:app --reload --port 8001
```

### Issue: Import errors
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: ONNX Runtime errors
**Solution**: Make sure `onnxruntime` is installed:
```bash
pip install onnxruntime
```

## 📝 Notes

- The model handles unknown categorical values by mapping them to an "Unknown" category
- Training uses GPU if available (automatically detected by PyTorch Lightning)
- The preprocessor ensures consistent feature ordering between training and inference
- Model checkpoints are saved automatically during training

## 📄 License

This project is for educational purposes. The dataset is from Kaggle's Playground Series.

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements!

---

**Happy Predicting! 💎**

