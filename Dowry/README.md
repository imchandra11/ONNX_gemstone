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




# Dummy Dowry Calculation Dataset (Awareness & Experimental Use Only)

## 📌 Overview

This repository contains a **synthetic (dummy) dataset** created for **machine learning experimentation, data analysis, and social awareness purposes only**.

The dataset simulates how **societal bias** (not logic, law, or ethics) often evaluates a groom’s “value” in the context of Indian marriages using superficial attributes such as salary, profession, age, and location.

⚠️ **Important:**  
This dataset does **NOT** promote, support, or justify the dowry system in any form.  
Dowry is illegal in India under the **Dowry Prohibition Act, 1961**.

---

## 🎯 Purpose of the Dataset

- To experiment with **regression and explainable AI models**
- To visualize **societal bias using data**
- To promote **awareness and self-reflection**
- To show how humans are unfairly reduced to numbers

If the predictions or patterns feel uncomfortable — that discomfort is intentional and meaningful.

---

## 🧠 Problem Statement

**Objective:**  
Predict a *society-perceived dowry amount* based on attributes often (wrongly) considered important in marriage negotiations.

This is **not a real valuation** of a human being.

---

## 📊 Dataset Structure

### Input Features

| Feature | Type | Description |
|------|------|------------|
| `age` | Integer | Age of groom (22–40 years) |
| `monthly_salary` | Integer | Monthly income in INR |
| `profession` | Categorical | Job category |
| `education_level` | Categorical | Highest education attained |
| `location` | Categorical | City tier |
| `home_status` | Categorical | Own or rented house |
| `family_wealth` | Categorical | Family economic background |
| `marital_status` | Categorical | Single / Married / Divorced |
| `marriage_type` | Categorical | Love or Arranged marriage |
| `government_job` | Binary | Government job indicator (0 or 1) |

### Target Variable

| Feature | Type | Description |
|------|------|------------|
| `dowry_amount_lakhs` | Float | Simulated dowry value in lakhs INR |

---

## 🧮 How the Target Is Generated

The `dowry_amount_lakhs` value is generated using **rule-based societal assumptions**, such as:

- Higher salary → higher perceived value
- Government jobs receive a significant premium
- Urban locations are valued more than rural
- Love marriages and remarriages receive penalties
- Education and property ownership increase value

These rules are **deliberately biased** to reflect social realities — not correctness.

---

## 🧪 Sample Data

```csv
age,monthly_salary,profession,education_level,location,home_status,family_wealth,marital_status,marriage_type,government_job,dowry_amount_lakhs
26,80000,Software Engineer,Masters,Tier-1,Own,Upper-Middle,Single,Arranged,0,28.5
30,45000,Government Officer,Bachelors,Tier-2,Own,Middle,Single,Arranged,1,32.0
28,120000,Doctor,Masters,Tier-1,Own,Upper,Single,Love,0,38.5
34,60000,Private Job,Bachelors,Tier-3,Rented,Middle,Divorced,Arranged,0,12.5
