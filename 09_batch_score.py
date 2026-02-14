import os
import joblib
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime

DATA_PATH = "data/gold/churn_features.parquet"
REASONS_PATH = "data/gold/reason_codes.parquet"
MODEL_PATH = "models/churn_model.joblib"
METRICS_PATH = "models/metrics.json"

GOLD_DIR = "data/gold"
DB_PATH = "warehouse.db"

os.makedirs(GOLD_DIR, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
reasons = pd.read_parquet(REASONS_PATH)
model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

threshold = float(metrics["chosen_threshold"])

# Features only
X = df.drop(columns=["customerID", "Churn"], errors="ignore")

# Predict probabilities
proba = model.predict_proba(X)[:, 1]

out = pd.DataFrame({
    "customerID": df["customerID"].values,
    "as_of_date": datetime.today().strftime("%Y-%m-%d"),
    "churn_proba": proba
})

# Risk bands (business-friendly)
out["risk_band"] = pd.cut(
    out["churn_proba"],
    bins=[-0.0001, threshold, 0.60, 1.0],
    labels=["Low", "Medium", "High"]
).astype(str)

# Join reason codes
out = out.merge(reasons, on="customerID", how="left")

# Sort by highest churn risk
out = out.sort_values("churn_proba", ascending=False).reset_index(drop=True)

# Save parquet output
out_path = f"{GOLD_DIR}/predictions_daily.parquet"
out.to_parquet(out_path, index=False)

# Save to SQLite (warehouse-like)
conn = sqlite3.connect(DB_PATH)
out.to_sql("predictions_daily", conn, if_exists="replace", index=False)
conn.close()

print("✅ Batch scoring completed")
print(f"Saved: {out_path}")
print(f"Saved: {DB_PATH} (table: predictions_daily)")
print("\nTop 10 high-risk customers:")
print(out.head(10)[["customerID", "churn_proba", "risk_band", "reason1", "reason2", "reason3"]])
