# Dowry Amount Prediction (Awareness-Only Project)

> ⚠️ **This project is for social awareness and experimentation only.**  
> Dowry is illegal in India under the Dowry Prohibition Act, 1961.  
> This model simulates societal bias and must not be used to justify or promote dowry.

## Overview
An end-to-end machine learning pipeline that predicts a simulated *dowry amount* (in lakhs INR) based on synthetic, biased features. It mirrors the diamond project structure: PyTorch Lightning for training, ONNX export for inference, and a FastAPI web UI.

## Dataset
- File: `dowry_dataset.csv`
- Target: `dowry_amount_lakhs` (float)
- Features:
  - Numeric: `age`, `monthly_salary`, `government_job` (0/1)
  - Categorical: `profession`, `education_level`, `location`, `home_status`, `family_wealth`, `marital_status`, `marriage_type`

## Project Structure
```
Dowry/
├── dowry_dataset.csv
├── train_dowry.py          # Training + ONNX export
├── models/                 # Generated artifacts after training
│   ├── dowry_regressor.onnx
│   ├── preprocessor.joblib
│   ├── metadata.json
│   └── dowry-regressor.ckpt
└── app/
    ├── main.py             # FastAPI app
    ├── templates/index.html
    └── static/styles.css
```

## Setup (reuse existing virtual env)
From repo root (env already created for first project):
```powershell
.\.venv\Scripts\Activate.ps1
cd Dowry
```

If needed, install dependencies (already listed in root `requirements.txt`):
```powershell
pip install -r ..\requirements.txt
```

## Train the model
```powershell
cd Dowry
python train_dowry.py --csv dowry_dataset.csv --output-dir models --epochs 30 --batch-size 256 --lr 1e-3
```
Artifacts saved to `Dowry/models/`:
- `dowry_regressor.onnx`
- `preprocessor.joblib`
- `metadata.json`
- Lightning checkpoint: `dowry-regressor.ckpt`

## Run the FastAPI app
```powershell
cd Dowry
uvicorn app.main:app --reload --port 8001
```
- UI: http://localhost:8001  
- API docs: http://localhost:8001/docs

## Form Inputs (UI)
- Age (int), Monthly Salary (INR)
- Profession: Software Engineer, Government Officer, Private Job, Doctor, Business Owner, Freelancer, CA
- Education Level: Diploma, Bachelors, Masters, PhD
- Location: Tier-1, Tier-2, Tier-3, Rural
- Home Status: Own, Rented
- Family Wealth: Lower, Middle, Upper-Middle, Upper
- Marital Status: Single, Married, Divorced
- Marriage Type: Arranged, Love
- Government Job: 0 or 1

## Notes & Ethics
- Synthetic, biased data; results are not real or endorsed.
- Do not deploy for production/real decisions.
- Purpose: awareness, experimentation, and demonstrating ML pipelines.

## Common Issues
- If ONNXRuntime DLL error: reinstall `onnxruntime==1.16.3` (already pinned).
- Ensure `.venv` is active before running training or app.

---
**Use responsibly. Dowry is illegal and unethical.**

