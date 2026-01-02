"""
FastAPI application for Gemstone Price Prediction
"""
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


CATEGORICAL_COLS = ["cut", "color", "clarity"]
NUMERIC_COLS = ["carat", "depth", "table", "x", "y", "z"]


class Preprocessor:
    """Preprocessor for inference matching training pipeline."""

    def __init__(self, payload: Dict):
        self.cat_levels: Dict[str, List[str]] = payload["cat_levels"]
        self.feature_names: List[str] = payload["feature_names"]
        self.scaler_mean = np.array(payload["scaler_mean"])
        self.scaler_scale = np.array(payload["scaler_scale"])

    def transform(self, row: Dict[str, str]) -> np.ndarray:
        """Transform input row to model features."""
        df = pd.DataFrame([row])
        for col in CATEGORICAL_COLS:
            allowed = set(self.cat_levels[col])
            val = df.at[0, col]
            if val not in allowed:
                df.at[0, col] = "Unknown"
            df[col] = pd.Categorical(df[col], categories=self.cat_levels[col])

        cat_df = pd.get_dummies(df[CATEGORICAL_COLS], prefix_sep="_", dummy_na=False)
        for col in self.feature_names:
            if col.startswith(tuple(CATEGORICAL_COLS)) and col not in cat_df.columns:
                cat_df[col] = 0
        cat_df = cat_df[[c for c in self.feature_names if c in cat_df.columns]]

        num_vals = df[NUMERIC_COLS].astype(float).values
        num_scaled = (num_vals - self.scaler_mean) / self.scaler_scale

        features = np.hstack([cat_df.values, num_scaled])
        ordered = pd.DataFrame(features, columns=self.feature_names)
        return ordered.astype(np.float32).values


def load_artifacts(model_dir: Path):
    """Load preprocessor and ONNX model."""
    # Load preprocessor
    preproc_path = model_dir / "preprocessor.joblib"
    if not preproc_path.exists():
        raise FileNotFoundError(
            f"preprocessor.joblib not found in {model_dir}. Run training first."
        )
    preproc_payload = joblib.load(preproc_path)
    preproc = Preprocessor(preproc_payload)

    # Load ONNX model
    onnx_path = model_dir / "gemstone_price_model.onnx"
    if not onnx_path.exists():
        # Try to get from metadata if it exists
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            onnx_path = model_dir / Path(metadata.get("onnx_model", "gemstone_price_model.onnx")).name
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found in {model_dir}. Run training first."
            )
    
    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    return preproc, session


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
preprocessor, onnx_session = load_artifacts(MODEL_DIR)

app = FastAPI(title="Gemstone Price Predictor")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render prediction form."""
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    carat: float = Form(...),
    cut: str = Form(...),
    color: str = Form(...),
    clarity: str = Form(...),
    depth: float = Form(...),
    table: float = Form(...),
    x: float = Form(...),
    y: float = Form(...),
    z: float = Form(...),
):
    """Predict gemstone price from form input."""
    row = {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
    }
    features = preprocessor.transform(row)
    inputs = {onnx_session.get_inputs()[0].name: features}
    pred = onnx_session.run(None, inputs)[0][0][0]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": float(pred), "form_values": row},
    )

