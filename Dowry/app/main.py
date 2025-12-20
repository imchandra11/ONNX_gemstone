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


CATEGORICAL_COLS = [
    "profession",
    "education_level",
    "location",
    "home_status",
    "family_wealth",
    "marital_status",
    "marriage_type",
]
NUMERIC_COLS = ["age", "monthly_salary", "government_job"]


class Preprocessor:
    def __init__(self, payload: Dict):
        self.cat_levels: Dict[str, List[str]] = payload["cat_levels"]
        self.feature_names: List[str] = payload["feature_names"]
        self.scaler_mean = np.array(payload["scaler_mean"])
        self.scaler_scale = np.array(payload["scaler_scale"])

    def transform(self, row: Dict[str, str]) -> np.ndarray:
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
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {model_dir}. Run train_dowry.py first."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    preproc_payload = joblib.load(model_dir / "preprocessor.joblib")
    preproc = Preprocessor(preproc_payload)

    onnx_path = model_dir / "dowry_regressor.onnx"
    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    return preproc, session


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
preprocessor, onnx_session = load_artifacts(MODEL_DIR)

app = FastAPI(title="Dowry Amount Predictor (Awareness Only)")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    monthly_salary: int = Form(...),
    profession: str = Form(...),
    education_level: str = Form(...),
    location: str = Form(...),
    home_status: str = Form(...),
    family_wealth: str = Form(...),
    marital_status: str = Form(...),
    marriage_type: str = Form(...),
    government_job: str = Form("0"),
):
    # Handle toggle switch: if checked, value is "1", otherwise "0" from hidden input
    gov_job_value = 1 if government_job == "1" else 0
    
    row = {
        "age": age,
        "monthly_salary": monthly_salary,
        "profession": profession,
        "education_level": education_level,
        "location": location,
        "home_status": home_status,
        "family_wealth": family_wealth,
        "marital_status": marital_status,
        "marriage_type": marriage_type,
        "government_job": gov_job_value,
    }
    features = preprocessor.transform(row)
    inputs = {onnx_session.get_inputs()[0].name: features}
    pred = onnx_session.run(None, inputs)[0][0][0]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": float(pred), "form_values": row},
    )

