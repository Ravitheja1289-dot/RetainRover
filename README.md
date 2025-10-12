# RetainRover — Run & Deploy Guide

This repository contains RetainRover (a.k.a. InsuraSense) — a Streamlit-based Machine Learning dashboard for predicting customer/employee retention (churn) and explaining predictions using SHAP and LIME.

This README focuses on how to run the project locally, run the Streamlit app, and where to find the live deployment.

Live demo: https://retainrover.streamlit.app/

---

## Contents of this repository

- `app.py` — Main Streamlit application (dashboard, model training, SHAP & LIME explanations).
- `datatraining/` — Training and sample data, training scripts, and model artifacts.
- `models/` — Saved model artifacts (pipeline, pickles) if available.
- `public/`, `scripts/`, `src/` — Frontend static files and React app used for a modern UI (optional).
- `requirements.txt` — Python dependencies required to run the Streamlit app.

## Quick links

- Live application: https://retainrover.streamlit.app/
- Run the Streamlit app locally: `streamlit run app.py`

---

## Running the Streamlit app (Windows)

These instructions assume you want to run the Streamlit Python app in `app.py`. The project supports Windows PowerShell and cmd.exe workflows.

Recommended Python version: 3.10 or 3.11

### 1) Create & activate a virtual environment (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 1b) Create & activate a virtual environment (cmd.exe)
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Generate or ensure sample data exists
If you don't have your own dataset, generate the sample dataset used for demos:

```powershell
python datatraining/generate_sample_data.py
```

This writes `datatraining/data/churn_data.csv` with sample rows used by the dashboard.

### 3) Run the Streamlit app

```powershell
streamlit run app.py
```

Visit http://localhost:8501 in your browser (Streamlit will usually open a browser tab automatically).

---

## How the app works (short)

- Load data (upload CSV or use sample data)
- Train a model (RandomForest / XGBoost if available)
- Generate predictions across dataset
- Explain individual predictions with SHAP (global/local) and LIME (local)
- Export predictions as CSV

## Files you will interact with

- `app.py` — main app file where UI, training, explainability, and download buttons live
- `datatraining/` — contains training scripts and `generate_sample_data.py` to create example CSVs
- `requirements.txt` — required packages for the Streamlit app

---

## Troubleshooting

- If Streamlit fails to start, make sure you activated the virtual environment and installed `requirements.txt`.
- If LIME or XGBoost aren't available, the app will fall back to SHAP or RandomForest; check console logs for warnings about unavailable packages.
- If uploads are large, the app limits to a maximum number of rows (see the Data Input tab in `app.py`).

## Deployments & Live demo

This repository is also deployed to Streamlit Cloud. Live demo URL:

https://retainrover.streamlit.app/

---

