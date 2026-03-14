# ⛈️ Thunderstorm Forecasting System (THFS)

> **Predicting Thunderstorm Occurrence from Atmospheric Indices with Machine Learning**
>
> An end-to-end MLOps system that takes 8 atmospheric stability indices and predicts whether a **Thunderstorm (TH) will occur** — with experiment tracking via MLflow, a FastAPI prediction backend, and a Streamlit web interface.

---

<div align="center">

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Model-RandomForest%20%7C%20KNN-orange)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📊 Project Slides

> **Want the visual overview first?** The deck covers business problem, architecture, experiments, model results, and API design in 12 slides.

👉 **[View the Project Presentation (PPTX)](https://docs.google.com/presentation/d/1ozwWWEWSbgGC0ARYrLIkAavACsq9K7Ct/edit?usp=sharing&ouid=117459468470211543781&rtpof=true&sd=true)**

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Business Problem](#1-business-problem) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [High-Level Architecture](#4-high-level-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Data & Features](#6-data--features) |
| 7 | [Experiments & Model Selection](#7-experiments--model-selection) |
| 8 | [Model Performance — Best Results](#8-model-performance--best-results) |
| 9 | [API & Web Application](#9-api--web-application) |
| 10 | [How to Replicate — Full Setup Guide](#10-how-to-replicate--full-setup-guide) |
| 11 | [Running the Application](#11-running-the-application) |
| 12 | [Deployment](#12-deployment) |
| 13 | [Business Applications & Other Domains](#13-business-applications--other-domains) |
| 14 | [How to Improve This Project](#14-how-to-improve-this-project) |
| 15 | [Troubleshooting](#15-troubleshooting) |
| 16 | [Glossary](#16-glossary) |

---

## 1. Business Problem

### What problem are we solving?

Thunderstorms cause significant economic and safety damage — flight diversions, disruption to outdoor operations, infrastructure damage, and risk to human life. Traditional meteorological forecasting relies on Numerical Weather Prediction (NWP) models that are computationally expensive, require specialist expertise, and produce probabilistic outputs that are difficult for operational teams to act on directly.

Atmospheric stability indices — calculated from radiosonde (weather balloon) soundings — are well-established proxies for convective instability. However, interpreting these indices in combination to produce a binary thunderstorm forecast requires significant domain expertise and manual effort.

Core pain points:

- ⚡ **High false alarm rate** — manual index thresholding produces too many false alarms, causing alert fatigue
- 🛫 **Aviation safety** — missed thunderstorm forecasts on flight routes create serious safety risks
- 🌍 **Operational disruption** — outdoor events, agriculture, and construction need reliable short-range TH predictions
- 🔬 **Reproducibility** — manual expert judgment is not reproducible or auditable

### What does THFS answer?

> *"Given 8 atmospheric stability index values measured at a weather station, will a thunderstorm occur at that location? And with what probability?"*

### Objectives

1. Build a binary classifier (TH = 0: no thunderstorm, TH = 1: thunderstorm) from 8 atmospheric indices
2. Handle severe class imbalance using **SMOTE** (thunderstorms are rare events)
3. Track all model experiments with **MLflow** — parameters, metrics, and model artifacts
4. Evaluate using **meteorological skill scores** (POD, FAR, HSS, CSI) alongside standard ML metrics
5. Serve predictions via a **FastAPI REST backend** with a **Streamlit** web frontend
6. Package for deployment on Render or AWS

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Task** | Binary classification: Thunderstorm occurrence (TH = 0 / 1) |
| **Input features** | 8 atmospheric stability indices (see Section 6) |
| **Dataset** | ~11,683 daily observations (1981–present, merged index + surface data) |
| **Class imbalance** | Addressed with SMOTE before training |
| **Models compared** | KNN, Random Forest, Decision Tree (all tracked in MLflow) |
| **Best model (deployed)** | KNN (`n_neighbors=3`, `weights='distance'`) |
| **Model format** | joblib-compressed `.pkl` (~5–10 MB) |
| **Experiment tracking** | MLflow (local `mlflow.db` + `mlruns/` artifacts) |
| **Backend API** | FastAPI on port 8000 (`uvicorn`) |
| **Frontend** | Streamlit (two versions: direct model load and API-connected) |
| **Package manager** | `uv` (ultra-fast Python package manager) |
| **Python version** | ≥ 3.13 |
| **Deployment target** | Render / AWS / Docker |

---

## 3. Tech Stack

### Complete Technology Map

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python ≥ 3.13 | Core language |
| **Package manager** | `uv` | Fast virtual environment and dependency management |
| **ML Models** | Scikit-learn (`KNeighborsClassifier`, `RandomForestClassifier`, `DecisionTreeClassifier`) | Classification models for TH prediction |
| **Class imbalance** | `imbalanced-learn` (SMOTE) | Oversamples minority (thunderstorm) class before training |
| **Experiment tracking** | MLflow 3.7.0 | Logs params, metrics, and model artifacts for all runs |
| **Gradient boosting** | XGBoost | Available for extended experiments (not in final deployed model) |
| **Data processing** | Pandas, NumPy | Data loading, feature engineering, train/test split |
| **Visualisation** | Matplotlib, Seaborn | EDA and performance plots in experiment notebook |
| **Model serialisation** | joblib | Compresses and saves/loads model (~5–10 MB) |
| **Schema validation** | Pydantic v2 | Validates all 8 API input fields via `WeatherInput` model |
| **REST API** | FastAPI 0.124.2 + Uvicorn | Serves `/predict` endpoint on port 8000 |
| **Web UI (direct)** | `app.py` (Streamlit) | Loads model directly — standalone inference without FastAPI |
| **Web UI (API-connected)** | `streamlit_app/ui.py` | Calls FastAPI backend via HTTP POST |
| **App layer** | `app/` package | Modular: `config.py`, `model_loader.py`, `predictor.py`, `schemas.py` |
| **Containerisation** | Docker | Ready for cloud deployment (Render / AWS) |
| **Experiment database** | SQLite (`mlflow.db`) | Stores MLflow experiment metadata locally |

---

## 4. High-Level Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│                                                                     │
│  [ index.csv ]    [ surface.csv ]                                   │
│  Radiosonde       Surface obs.                                      │
│  11,883 rows      14,397 rows                                       │
│        │                │                                           │
│        └────────┬───────┘                                           │
│                 ▼                                                   │
│  [ merged_df_all12k_combined.csv ]  — 11,682 rows                  │
│    8 features + TH target (0/1)                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT LAYER                                  │
│                                                                     │
│  experiment.ipynb → SMOTE → train/test split                        │
│                           │                                         │
│       ┌───────────────────┼──────────────────────┐                 │
│       ▼                   ▼                      ▼                  │
│   [KNN run]       [Random Forest run]  [Decision Tree run]          │
│   n_neighbors=3   n_estimators=100     max_depth=20                 │
│       │                   │                      │                  │
│       └───────────────────┴──────────────────────┘                 │
│                           │                                         │
│                  [ MLflow tracking ]                                │
│                  mlflow.db + mlruns/                                │
│                           │                                         │
│                  [ Best model saved ]                               │
│                  models/KNN_best_model.pkl                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                                  │
│                                                                     │
│  Mode A — Direct:                                                   │
│  streamlit run app.py  →  loads KNN_best_model.pkl directly        │
│                                                                     │
│  Mode B — Decoupled:                                                │
│  uvicorn api.main:app --port 8000  (FastAPI backend)               │
│        +                                                            │
│  streamlit run streamlit_app/ui.py  (UI calls POST /predict)        │
│                                                                     │
│  Input: 8 atmospheric indices → Output: TH (0/1) + probability     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data & Artifact Flow

| Step | Input | Output |
|------|-------|--------|
| Data prep | `index.csv` + `surface.csv` | `merged_df_all12k_combined.csv` |
| SMOTE | Imbalanced training data | Balanced synthetic training set |
| MLflow experiments | 3 algorithms × multiple runs | Params + metrics logged to `mlflow.db` |
| Model selection | Best MLflow run | `models/KNN_best_model.pkl` |
| FastAPI | HTTP POST `/predict` JSON | `{"prediction": 0/1, "probability": float}` |
| Streamlit UI | 8 form inputs | Prediction + probability displayed |

---

## 5. Repository Structure

```
thunderstorm-forecasting-project-full/
│
├── app/                            # Core application package
│   ├── __init__.py
│   ├── config.py                   # MODEL_PATH = "models/KNN_best_model.pkl"
│   ├── model_loader.py             # Singleton: loads model once at startup via joblib
│   ├── predictor.py                # predict_weather(features) → {prediction, probability}
│   └── schemas.py                  # Pydantic WeatherInput — validates all 8 input fields
│
├── api/
│   └── main.py                     # FastAPI app: GET / health + POST /predict
│
├── streamlit_app/
│   └── ui.py                       # Streamlit UI — calls FastAPI at http://localhost:8000/predict
│
├── app.py                          # Standalone Streamlit app — loads model directly (no FastAPI)
│
├── experiments/
│   ├── experiment.ipynb            # Full ML experiment: EDA, SMOTE, KNN/RF/DT training, MLflow logging
│   ├── mlflow.db                   # SQLite MLflow tracking database
│   ├── mlruns/                     # MLflow run artifacts (params, metrics, model PKLs)
│   │   ├── 1/models/               # 3 registered model versions
│   │   └── 162491754287900972/     # 9 experiment runs (3 KNN, 3 RF, 3 DT)
│   └── KNN_best_model.pkl          # Best model copy in experiments/
│
├── models/
│   └── KNN_best_model.pkl          # Production model — loaded by app/ package
│
├── data/
│   ├── raw/
│   │   ├── index.csv               # Atmospheric stability indices (11,883 rows)
│   │   └── surface.csv             # Surface TH observations (14,397 rows)
│   └── processed/
│       └── merged_df_all12k_combined.csv  # Final merged dataset (11,682 rows, 8 features + TH)
│
├── main.py                         # uv project entry point (placeholder)
├── pyproject.toml                  # uv project config (Python ≥ 3.13)
├── requirements.txt                # Production deps: streamlit, joblib, scikit-learn, numpy, pandas
├── local-requirements.txt          # Full dev deps + MLflow, XGBoost, FastAPI, gunicorn, etc.
└── .python-version                 # Python version pin for uv
```

---

## 6. Data & Features

### Dataset Overview

| File | Source | Rows | Description |
|------|--------|------|-------------|
| `data/raw/index.csv` | Radiosonde soundings | 11,883 | Daily atmospheric stability indices per year/month/day |
| `data/raw/surface.csv` | Surface observations | 14,397 | Daily TH occurrence (0/1) per year/month/day |
| `data/processed/merged_df_all12k_combined.csv` | Merged | 11,682 | 8 input features + TH target, ready for training |

The dataset spans from **1981** to present — over 40 years of daily atmospheric observations at a single weather station.

### Input Features (8 Atmospheric Stability Indices)

| Feature | Description | Meteorological Significance |
|---------|-------------|----------------------------|
| **SWEAT Index** | Severe Weather ThrEAT index | Combines wind shear, moisture, and instability — high SWEAT indicates severe thunderstorm risk |
| **K Index** | K Index (George, 1960) | Measures moisture content and atmospheric lapse rate at mid-levels; >35 = high TH risk |
| **Totals Totals Index** | Cross Totals + Vertical Totals | Combined temperature lapse rate measure; >55 = significant thunderstorm risk |
| **Environmental_Stability** | Derived stability composite | Combined measure of atmospheric layer stability |
| **Moisture_Indices** | Moisture availability | Precipitable water / dewpoint depression composite |
| **Convective_Potential** | CAPE-derived measure | Convective Available Potential Energy — energy available for storm development |
| **Temperature_Pressure** | 1000–500 hPa thickness | Temperature structure of the troposphere; linked to airmass stability |
| **Moisture_Temperature_Profiles** | PLCL / boundary layer | Lifted Condensation Level pressure — height where air parcel becomes saturated |

### Target Variable

| Column | Values | Description |
|--------|--------|-------------|
| `TH` | 0 / 1 | Thunderstorm occurrence: 0 = no TH, 1 = TH occurred on that day |

### Class Imbalance

Thunderstorms are rare events — the `TH=1` class is significantly underrepresented in the dataset. **SMOTE (Synthetic Minority Oversampling Technique)** is applied during training to create synthetic minority-class samples, ensuring the model learns to detect thunderstorms rather than defaulting to always predicting no-TH.

---

## 7. Experiments & Model Selection

All experiments were run in `experiments/experiment.ipynb` and tracked in MLflow (`mlflow.db`). Three algorithm families were compared across multiple runs:

### Experiment Results (All MLflow Runs)

| Run | Algorithm | n_estimators / n_neighbors | max_depth | Accuracy | F1 | POD | FAR | HSS | CSI |
|-----|-----------|--------------------------|-----------|----------|-----|-----|-----|-----|-----|
| KNN (run 1) | KNeighborsClassifier | n_neighbors=3, weights='distance' | — | 0.8290 | 0.8492 | **0.9694** | 0.3097 | 0.6586 | 0.7379 |
| KNN (run 2) | KNeighborsClassifier | n_neighbors=3, weights='distance' | — | 0.8244 | 0.8442 | 0.9576 | 0.3070 | 0.6495 | 0.7304 |
| KNN (run 3) | KNeighborsClassifier | n_neighbors=3, weights='distance' | — | 0.8290 | 0.8492 | **0.9694** | 0.3097 | 0.6586 | 0.7379 |
| RF (run 1) | RandomForestClassifier | n_estimators=100 | None | 0.8135 | 0.8281 | 0.9044 | 0.2762 | 0.6274 | 0.7067 |
| RF (run 2) | RandomForestClassifier | n_estimators=100 | None | 0.8188 | 0.8330 | 0.9092 | **0.2704** | 0.6381 | 0.7137 |
| RF (run 3) | RandomForestClassifier | n_estimators=100 | None | 0.8188 | 0.8330 | 0.9092 | **0.2704** | 0.6381 | 0.7137 |
| DT (run 1) | DecisionTreeClassifier | — | max_depth=20 | 0.7607 | 0.7738 | 0.8238 | 0.3017 | 0.5217 | 0.6310 |
| DT (run 2) | DecisionTreeClassifier | — | max_depth=20 | 0.7516 | 0.7633 | 0.8061 | 0.3022 | 0.5035 | 0.6172 |
| DT (run 3) | DecisionTreeClassifier | — | max_depth=20 | 0.7607 | 0.7738 | 0.8238 | 0.3017 | 0.5217 | 0.6310 |

### Why KNN Won

KNN (`n_neighbors=3`, `weights='distance'`) was selected as the best model and deployed as `KNN_best_model.pkl` based on:

| Criterion | KNN | Random Forest | Decision Tree |
|-----------|-----|---------------|---------------|
| **Accuracy** | 0.829 ✅ | 0.819 | 0.761 |
| **F1 Score** | **0.849** ✅ | 0.833 | 0.774 |
| **POD (recall)** | **0.969** ✅ | 0.909 | 0.824 |
| **FAR** | 0.310 | **0.270** | 0.302 |
| **HSS** | **0.659** ✅ | 0.638 | 0.522 |
| **CSI** | **0.738** ✅ | 0.714 | 0.631 |

**For thunderstorm forecasting, maximising POD (Probability of Detection) is the priority** — missing a thunderstorm is more costly than a false alarm. KNN's POD of 0.969 means it detects 96.9% of actual thunderstorm days. While its FAR is slightly higher than Random Forest, the superior POD and HSS make it the operational choice.

---

## 8. Model Performance — Best Results

### Best Model: KNN (`n_neighbors=3`, `weights='distance'`)

```json
{
  "accuracy":  0.8290,
  "f1_score":  0.8492,
  "precision": 0.7555,
  "recall":    0.9694,
  "pod":       0.9694,
  "far":       0.3097,
  "hss":       0.6586,
  "csi":       0.7379
}
```

### Metric Interpretations

| Metric | Value | Plain-English Meaning |
|--------|-------|----------------------|
| **Accuracy** | 82.9% | Overall correct predictions across both classes |
| **F1 Score** | 0.849 | Harmonic mean of precision and recall — strong balanced performance |
| **Precision** | 0.756 | When the model predicts a thunderstorm, it's correct 75.6% of the time |
| **Recall / POD** | 0.969 | The model catches 96.9% of all actual thunderstorm events — critical for safety |
| **FAR** | 0.310 | 31% of thunderstorm predictions are false alarms |
| **HSS** | 0.659 | 65.9% better than a random forecast — strong skill |
| **CSI** | 0.738 | Considers hits, misses, and false alarms together — 0.74 is excellent for TH forecasting |

### Model Configuration

```python
KNeighborsClassifier(
    n_neighbors = 3,
    weights     = 'distance'   # closer neighbours weighted more
)
# Trained on SMOTE-balanced data
# Compressed with joblib, ~5–10 MB
```

**Why `weights='distance'`?** Standard KNN treats all K neighbours equally. With `weights='distance'`, the nearest neighbour has more influence than a slightly farther one — important for atmospheric data where small index differences carry physical meaning.

---

## 9. API & Web Application

### Dual Serving Architecture

The project supports **two modes** of operation:

**Mode A — Standalone Streamlit (recommended for demos):**
```
streamlit run app.py  →  loads KNN_best_model.pkl directly  →  shows prediction + probability
```

**Mode B — Decoupled FastAPI + Streamlit:**
```
uvicorn api.main:app --port 8000        (backend)
+
streamlit run streamlit_app/ui.py       (frontend calls POST /predict)
```

---

### FastAPI Backend (`api/main.py`)

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Health check — returns `{"message": "Weather Prediction API is running"}` |
| `POST` | `/predict` | Accepts `WeatherInput` JSON → returns `{prediction: int, probability: float}` |

**Auto-generated API docs** available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

### Input Schema (`app/schemas.py` — Pydantic v2)

```python
class WeatherInput(BaseModel):
    SWEAT_index:                    float
    K_index:                        float
    Totals_totals_index:            float
    Environmental_Stability:        float
    Moisture_Indices:               float
    Convective_Potential:           float
    Temperature_Pressure:           float
    Moisture_Temperature_Profiles:  float
```

### Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SWEAT_index": 280.5,
    "K_index": 38.2,
    "Totals_totals_index": 52.1,
    "Environmental_Stability": 28.4,
    "Moisture_Indices": 34.6,
    "Convective_Potential": 1200.0,
    "Temperature_Pressure": 5580,
    "Moisture_Temperature_Profiles": 924.5
  }'
```

**Example response:**
```json
{"prediction": 1, "probability": 0.8742}
```

### Application Package (`app/`)

| Module | Role |
|--------|------|
| `config.py` | `MODEL_PATH = "models/KNN_best_model.pkl"` — single source of truth for model location |
| `model_loader.py` | Singleton pattern: loads model once at startup via `joblib.load()` |
| `predictor.py` | `predict_weather(features)` — assembles DataFrame with correct column names, calls model, returns prediction + probability |
| `schemas.py` | Pydantic `WeatherInput` — validates all 8 fields are present and typed correctly before prediction |

---

## 10. How to Replicate — Full Setup Guide

### Prerequisites

- Python ≥ 3.13
- `uv` package manager ([install here](https://docs.astral.sh/uv/getting-started/installation/))
- Git

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sahatanmoyofficial/Thunderstorm-Forecasting.git
cd thunderstorm-forecasting-project-full
```

---

### Step 2 — Set Up Environment with uv

```bash
# Initialise uv (if not already done)
uv init

# Create virtual environment
uv venv thenv

# Activate
thenv\Scripts\activate          # Windows
source thenv/bin/activate       # Linux/Mac
```

---

### Step 3 — Install Dependencies

```bash
# Production only (Streamlit + model inference)
uv pip install -r requirements.txt

# Full dev environment (MLflow, XGBoost, FastAPI, notebooks)
uv pip install -r local-requirements.txt
```

---

### Step 4 — Verify Model File

The trained model is already included in the repository:

```bash
ls models/KNN_best_model.pkl   # Should exist (~5–10 MB)
```

No training is required to run the application.

---

### Step 5 — Review Experiments (Optional)

View all tracked MLflow runs:

```bash
cd experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 to browse all 9 experiment runs
```

---

## 11. Running the Application

### Mode A — Standalone Streamlit (simplest)

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

Fill in the 8 atmospheric index values and click **Predict** to get the thunderstorm classification and probability.

---

### Mode B — FastAPI + Streamlit (production pattern)

**Terminal 1 — Start FastAPI backend:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

**Terminal 2 — Start Streamlit frontend:**
```bash
streamlit run streamlit_app/ui.py
# Opens at http://localhost:8501
# Calls http://localhost:8000/predict
```

---

### Quick curl Test

```bash
# Health check
curl http://localhost:8000/

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"SWEAT_index":280.5,"K_index":38.2,"Totals_totals_index":52.1,"Environmental_Stability":28.4,"Moisture_Indices":34.6,"Convective_Potential":1200.0,"Temperature_Pressure":5580,"Moisture_Temperature_Profiles":924.5}'
```

---

## 12. Deployment

### Render (Recommended for quick deployment)

```bash
# Procfile (create in root)
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Push to GitHub → Connect to Render → Auto-deploy on every push. The Streamlit frontend can be deployed as a separate Render service pointing at the FastAPI service URL.

### Docker

```dockerfile
# Example Dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t thfs:latest .
docker run -d -p 8000:8000 thfs:latest
```

### AWS EC2

```bash
# On EC2 Ubuntu instance
sudo apt-get update && curl -fsSL https://get.docker.com | sh
git clone <your-repo> && cd thunderstorm-forecasting-project-full
docker build -t thfs:latest .
docker run -d -p 8000:8000 thfs:latest
# Open port 8000 in EC2 Security Group
```

> ⚠️ **Note:** If deploying the Streamlit UI separately, update `API_URL` in `streamlit_app/ui.py` from `http://localhost:8000/predict` to the deployed FastAPI URL.

---

## 13. Business Applications & Other Domains

### Primary Use Case — Operational Thunderstorm Forecasting

| User | Value Delivered |
|------|----------------|
| **Meteorologists** | Rapid binary TH forecast from sounding data — augments NWP model output |
| **Aviation operations** | Automated TH flag for flight route planning and ground operations |
| **Agriculture** | Pre-event warning for crop protection, irrigation scheduling, field operations |
| **Construction & events** | Real-time TH risk score for outdoor site safety planning |
| **Emergency management** | Early warning system integration for disaster preparedness |
| **Insurance** | Actuarial risk scoring for weather-related claims modelling |

### Adjacent Domains (Same Classification Pattern)

| Domain | Analogous Problem | Adaptation Needed |
|--------|-----------------|-------------------|
| **Fog forecasting** | Predict radiation fog from surface obs | Replace stability indices with visibility/dewpoint features |
| **Hail prediction** | Binary hail occurrence from CAPE/shear | Add wind shear and storm motion features |
| **Flash flood risk** | Binary flood event from rainfall/soil | Replace atmospheric with hydrological indices |
| **Air quality alerts** | Pollutant exceedance from met conditions | Replace TH with PM2.5 / ozone threshold flags |
| **Renewable energy** | Predict wind/solar resource availability | Use similar atmospheric profiling approach |
| **Sports & events** | Game cancellation risk model | Lightweight TH probability for venue operations |

---

## 14. How to Improve This Project

### 🧠 Model Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Tune KNN properly** | 🔴 High | Grid search over `n_neighbors` (1–20), `metric` (euclidean, manhattan, minkowski), `weights` — current runs all use the same params |
| **Random Forest tuning** | 🔴 High | RF had lower FAR (0.27) than KNN (0.31) — a tuned RF with HPT may beat KNN on the combined metric |
| **XGBoost / LightGBM** | 🔴 High | `local-requirements.txt` includes XGBoost but no XGB experiments were logged — add and compare |
| **Temporal cross-validation** | 🔴 High | Daily weather data is time-series — use `TimeSeriesSplit` instead of random split to avoid data leakage |
| **Feature engineering** | 🟡 Medium | Add interaction terms (SWEAT × K Index), lagged features (yesterday's indices), seasonal dummies (month) |
| **Calibration** | 🟡 Medium | Use `CalibratedClassifierCV` to improve probability outputs — important for operational use |

### 🏗️ MLOps & Engineering Improvements

| Area | Recommendation |
|------|---------------|
| **DagsHub integration** | `local-requirements.txt` includes `dagshub` — connect MLflow to DagsHub for remote experiment tracking and team collaboration |
| **GitHub Actions CI/CD** | Automate model retraining, validation, and deployment on data updates |
| **Model versioning** | Use MLflow Model Registry to manage staging/production model lifecycle |
| **FastAPI config management** | Move `API_URL` in `streamlit_app/ui.py` to environment variable — currently hardcoded |
| **Separate feature names** | `config.py` currently only holds `MODEL_PATH` — move `FEATURE_COLUMNS` from `predictor.py` to `config.py` |
| **Add `/health` endpoint** | Replace root `/` with a proper `/health` endpoint that also checks model is loaded |
| **Logging** | Add structured logging (Python `logging` / loguru) to FastAPI for production observability |
| **Unit tests** | Add `pytest` tests for `predict_weather()`, schema validation, and model loading |

### 📦 Product Improvements

- Display a **meteorological context panel** — show what the input index values mean (e.g. "K Index > 35 = High TH risk")
- Add **threshold guidance** — highlight which input fields are in the dangerous range
- Show **feature importance** from the Random Forest to explain which indices drove the prediction
- Add **time-series view** — upload a CSV of daily readings and get a forecast chart
- Integrate **sounding data APIs** to auto-populate index values from a weather station ID

---

## 15. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `ModuleNotFoundError: app` | Run from the project root directory, not from inside `api/` or `app/` |
| `FileNotFoundError: models/KNN_best_model.pkl` | Ensure `models/KNN_best_model.pkl` exists; it should be in the repo |
| Streamlit UI shows "API error" | FastAPI backend must be running first: `uvicorn api.main:app --port 8000` |
| `uvicorn` not found | Install with `uv pip install -r local-requirements.txt` |
| MLflow UI is empty | Run `mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db` from repo root |
| `ValueError: feature names mismatch` | `predictor.py` column names must exactly match training data — check `FEATURE_COLUMNS` list |
| `uv` command not found | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Port 8000 already in use | `lsof -ti:8000 \| xargs kill -9` or change port in the uvicorn command |
| Pydantic `ValidationError` | All 8 fields must be provided as floats — check your JSON payload |
| Probability is `null` | Only occurs if model lacks `predict_proba` — KNN does have it; this shouldn't happen |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **TH** | Thunderstorm — the binary target variable (0 = no thunderstorm, 1 = thunderstorm) |
| **SWEAT Index** | Severe Weather ThrEAT Index — combines wind speed, shear, moisture, and instability for severe storm risk |
| **K Index** | Atmospheric stability index measuring mid-tropospheric moisture and lapse rate (>35 = high TH risk) |
| **Totals Totals Index** | Sum of Cross Totals and Vertical Totals — temperature lapse measure; >55 = TH risk |
| **CAPE** | Convective Available Potential Energy — atmospheric energy available for storm updrafts |
| **PLCL** | Pressure at Lifted Condensation Level — height at which a rising air parcel becomes saturated |
| **POD** | Probability of Detection — fraction of actual thunderstorm events correctly predicted |
| **FAR** | False Alarm Rate — fraction of predicted thunderstorms that did not occur |
| **HSS** | Heidke Skill Score — measures improvement over random chance; 0 = no skill, 1 = perfect |
| **CSI** | Critical Success Index — combined measure of hits, misses, and false alarms; used operationally |
| **SMOTE** | Synthetic Minority Oversampling Technique — creates synthetic minority (TH=1) samples to address class imbalance |
| **Radiosonde** | Weather balloon instrument that measures temperature, humidity, and wind at various altitudes |
| **Stability Index** | Derived metric calculated from upper-air temperature and moisture profiles indicating convective instability |
| **joblib** | Python serialisation library optimised for NumPy arrays and large ML models — used here with compression |
| **Pydantic** | Python data validation library — ensures API inputs are correctly typed before reaching the model |
| **uvicorn** | ASGI server for running FastAPI applications in production |
| **uv** | Ultra-fast Python package and virtual environment manager (replacement for pip/venv/conda) |
| **MLflow** | Open-source MLOps platform for tracking experiments, parameters, metrics, and model artifacts |
| **DagsHub** | Remote MLflow tracking and Git hosting platform for ML projects — supported in this project |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com

---
