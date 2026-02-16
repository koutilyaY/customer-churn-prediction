# 🧠 Customer Churn Intelligence Platform

### Production-Grade End-to-End Machine Learning System for Predicting, Explaining, and Operationalizing Customer Churn Risk

---

## 🚀 Overview

Customer churn represents one of the largest hidden revenue losses in subscription-based industries. Most organizations struggle not only to predict churn but also to understand *why* customers leave and how to prioritize retention actions.

This project delivers a **production-style machine learning platform** that enables organizations to:

• Predict churn risk at scale  
• Generate interpretable reason codes for each customer  
• Optimize retention targeting using ROI-driven thresholds  
• Support enterprise analytics workflows using a medallion data architecture  

---

## 💼 Business Impact

| Capability | Business Value |
|------------|----------------|
| Churn Prediction | Early identification of at-risk customers |
| Explainable AI | Actionable retention insights |
| ROI Threshold Optimization | Reduces unnecessary marketing spend |
| Risk Segmentation | Enables targeted intervention strategies |
| Production Pipeline | Supports enterprise deployment readiness |

---

## 🏗️ Solution Architecture

```text
              ┌──────────────────────────┐
              │   Raw Telco Data (CSV)   │
              └──────────────┬───────────┘
                             ▼
              ┌──────────────────────────┐
              │     Bronze Layer         │
              │   Raw → Parquet Storage  │
              └──────────────┬───────────┘
                             ▼
              ┌──────────────────────────┐
              │     Silver Layer         │
              │   Data Cleaning & ETL    │
              └──────────────┬───────────┘
                             ▼
              ┌──────────────────────────┐
              │      Gold Layer          │
              │ Feature Engineering Hub  │
              └──────────────┬───────────┘
                             ▼
              ┌──────────────────────────┐
              │   ML Training Pipeline   │
              │  Model Calibration + ROI │
              └──────────────┬───────────┘
                             ▼
              ┌──────────────────────────┐
              │   Explainable AI Layer   │
              │   SHAP Reason Codes      │
              └──────────────┬───────────┘
                             ▼
              ┌──────────────────────────┐
              │   Batch Scoring Engine   │
              │  Risk Band Segmentation  │
              └──────────────────────────┘
```

---

## ⚙️ Key Engineering Highlights

### 🏗️ Data Architecture
• Medallion pipeline (Bronze → Silver → Gold)  
• Parquet-based optimized storage  
• Enterprise-style data modeling  

### 🤖 Machine Learning
• Logistic Regression baseline  
• Calibrated Gradient Boosting champion model  
• Probability reliability calibration  

### 📊 Explainability
• SHAP global feature importance  
• Per-customer churn reason codes  
• Model transparency for business users  

### 📈 Decision Optimization
• ROI-based threshold selection  
• Risk band classification (Low / Medium / High)  

---

## 📊 Model Performance

| Metric | Score |
|--------|------|
| ROC-AUC | **0.73** |
| PR-AUC | **0.75** |
| Calibration | High |
| Business Threshold | ROI-Optimized |

---

## 🔄 End-to-End Pipeline Workflow

```text
Ingest → Transform → Feature Engineering → Train → Explain → Score
```

### Detailed Steps

1️⃣ Data ingestion into Bronze storage  
2️⃣ Cleaning and transformation into Silver tables  
3️⃣ Feature engineering to build Gold dataset  
4️⃣ Model training and calibration  
5️⃣ Explainability using SHAP  
6️⃣ Batch scoring with churn risk bands  

---

## 🛠️ Tech Stack

### Data Engineering
Python • Pandas • Parquet

### Machine Learning
Scikit-learn • Gradient Boosting • Model Calibration

### Explainable AI
SHAP

---

## 📂 Project Structure

```
src/
├── ingest/
│   └── 01_ingest_telco.py
├── transform/
│   └── 02_bronze_to_silver.py
├── features/
│   └── 03_build_features.py
├── models/
│   └── 06_train_models.py
├── explain/
│   └── 08_shap_explanations.py
└── scoring/
    └── 09_batch_score.py
```

---

## 🧪 Real-World Applications

• Telecom churn prediction  
• Banking customer attrition modeling  
• SaaS subscription analytics  
• Insurance renewal risk prediction  

---

## 👨‍💻 Author

**Koutilya Yenumula**  
Data Engineer | Machine Learning Engineer  

---

## 📈 Future Enhancements

• Real-time churn prediction API  
• MLOps automation with CI/CD  
• Automated model monitoring  
• Cloud deployment pipeline  

---


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
