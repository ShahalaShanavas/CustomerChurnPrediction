# Customer Churn Prediction
End-to-end telecom **customer churn prediction** using **LightGBM, MLflow, FastAPI, and Gradio**.

## 🚀 Features
* Data validation, preprocessing, and feature engineering
* LightGBM classification with hyperparameter tuning
* MLflow experiment tracking
* Reproducible preprocessing and feature schema
* FastAPI REST API
* Gradio prediction UI
* Local model and preprocessing artifacts

## 🛠️ Tech Stack
Python 3.10+ · Pandas · Scikit-learn · LightGBM · Great Expectations · MLflow · FastAPI · Gradio

## 🔄 Workflow
```text
Raw Data → Validation → Preprocessing → Feature Engineering
→ Tuning → LightGBM → MLflow → Artifacts → Inference
→ FastAPI + Gradio → Churn Prediction
```

## 📊 Dataset
**Telco Customer Churn Dataset**
* Input: `data/raw/TelcoCustomerChurn.csv`
* Target: `Churn`
* Includes customer demographics, services, contract details, tenure, and charges.

> ⚠️ **Note:** The `data/` folder is not tracked in this repository (see `.gitignore`). Download the dataset from [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it at `data/raw/TelcoCustomerChurn.csv` before running the pipeline.

## ⚙️ Installation
```bash
git clone https://github.com/ShahalaShanavas/CustomerChurnPrediction.git
cd CustomerChurnPrediction
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Train the Model
```bash
python scripts/run_pipeline_final.py \
  --input data/raw/TelcoCustomerChurn.csv \
  --target Churn \
  --threshold 0.30
```
Generated model artifacts are stored in `artifacts/`.

> ⚠️ **Note:** Trained model artifacts (`*.pkl`) are not tracked in this repository. Run the training pipeline above to regenerate them locally before starting the API.

## 📈 MLflow
```bash
mlflow ui --backend-store-uri "sqlite:///./mlflow.db"
```
Open `http://127.0.0.1:5000`.

## 🌐 Run the API
```bash
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```
Endpoints:
| Method | Endpoint   | Description      |
| ------ | ---------- | ---------------- |
| GET    | `/`        | Health check     |
| POST   | `/predict` | Churn prediction |
| GET    | `/docs`    | Swagger UI       |
| GET    | `/ui`      | Gradio interface |

API documentation: `http://127.0.0.1:8000/docs`
Gradio UI: `http://127.0.0.1:8000/ui`

## 📦 Model Artifacts
* `best_params.json` — best hyperparameters
* `preprocessing.pkl` — preprocessing configuration
* `lightgbm_model.pkl` — trained model
* `feature_columns.json` — expected feature schema/order

## 🎯 Prediction Threshold
The default example uses `0.30`:
```text
Probability >= 0.30 → Churn
Probability < 0.30  → No Churn
```
The threshold can be adjusted based on business requirements.

## 👤 Maintainer
**Shahala Shanavas**
Repository: `ShahalaShanavas/CustomerChurnPrediction`

## 📜 License
No explicit open-source license is currently included.