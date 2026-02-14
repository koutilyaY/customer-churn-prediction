import pandas as pd
import joblib
import shap
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data/gold/churn_features.parquet"
MODEL_PATH = "models/churn_model.joblib"
REPORTS_DIR = "reports"
GOLD_DIR = "data/gold"

os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"], errors="ignore")

# Load calibrated model pipeline
calibrated_model = joblib.load(MODEL_PATH)

# Access underlying trained pipeline
pipeline = calibrated_model.calibrated_classifiers_[0].estimator

preprocess = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

# Transform features
X_processed = preprocess.transform(X)

# SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_processed)

# ---------- GLOBAL IMPORTANCE ----------
shap.summary_plot(shap_values, show=False)
plt.savefig(f"{REPORTS_DIR}/shap_summary.png", bbox_inches="tight")

# ---------- REASON CODES ----------
feature_names = preprocess.get_feature_names_out()

reason_rows = []

for i in range(len(df)):
    vals = shap_values.values[i]
    top_idx = np.argsort(np.abs(vals))[-3:][::-1]

    reasons = [feature_names[j] for j in top_idx]

    reason_rows.append({
        "customerID": df.iloc[i]["customerID"],
        "reason1": reasons[0],
        "reason2": reasons[1],
        "reason3": reasons[2]
    })

reason_df = pd.DataFrame(reason_rows)
reason_df.to_parquet(f"{GOLD_DIR}/reason_codes.parquet", index=False)

print("✅ SHAP explainability completed")
print("Saved:")
print(" - reports/shap_summary.png")
print(" - data/gold/reason_codes.parquet")
