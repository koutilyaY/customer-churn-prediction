# Customer Churn Prediction Intelligence Platform

End-to-end churn prediction project using the Telco Customer Churn dataset, built with a production-style pipeline (Bronze → Silver → Gold), calibrated ML modeling, explainable AI (SHAP), and batch scoring.

## Business Goal
Predict customers likely to churn and generate interpretable “reason codes” to support targeted retention actions.

## Key Results
- ROC-AUC: ~0.73 (test)
- PR-AUC: ~0.75 (test)
- ROI-based decision threshold selection for churn targeting

## Tech Stack
Python, Pandas, Scikit-learn, SHAP, Parquet

## Pipeline
1. **Ingest (Bronze)**: Load raw Telco CSV → Parquet
2. **Transform (Silver)**: Clean + split into enterprise-style tables
3. **Features (Gold)**: Build modeling dataset with engineered features
4. **Train**: Baseline Logistic Regression + calibrated Gradient Boosting champion
5. **Explain**: SHAP global importance + per-customer top reason codes
6. **Score**: Batch predictions with risk bands (Low/Medium/High)

## Project Structure
- `01_ingest_telco.py` – raw → bronze
- `02_bronze_to_silver.py` – bronze → silver (customers/services/billing/labels)
- `03_build_features.py` – silver → gold feature table
- `06_train_models.py` – training + calibration + ROI threshold selection
- `08_shap_explanations.py` – explainability + reason codes
- `09_batch_score.py` – batch scoring output

## How to Run (Local)
> Place the dataset file in `data/raw/` as:
> `WA_Fn-UseC_-Telco-Customer-Churn.csv`

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python 01_ingest_telco.py
python 02_bronze_to_silver.py
python 03_build_features.py
python 06_train_models.py
python 08_shap_explanations.py
python 09_batch_score.py
