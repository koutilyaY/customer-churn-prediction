import pandas as pd
import os
from glob import glob

BRONZE_DIR = "data/bronze"
SILVER_DIR = "data/silver"

os.makedirs(SILVER_DIR, exist_ok=True)

# Load latest bronze parquet
paths = sorted(glob(f"{BRONZE_DIR}/dt=*/telco_raw.parquet"))
if not paths:
    raise FileNotFoundError("No bronze parquet found. Run ingestion first.")
latest = paths[-1]

df = pd.read_parquet(latest)

# --- basic cleaning ---
# TotalCharges sometimes has blanks -> already coerced to NaN in bronze
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Ensure correct types
df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

# -------------------------
# 1) customers (dimension)
# -------------------------
customers = df[
    ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure"]
].drop_duplicates(subset=["customerID"])

customers.to_parquet(f"{SILVER_DIR}/customers.parquet", index=False)

# -------------------------
# 2) services (dimension)
# -------------------------
services = df[
    [
        "customerID",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
]

services.to_parquet(f"{SILVER_DIR}/services.parquet", index=False)

# -------------------------
# 3) billing (fact)
# -------------------------
billing = df[
    ["customerID", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
]

billing.to_parquet(f"{SILVER_DIR}/billing.parquet", index=False)

# -------------------------
# 4) churn_labels (label table)
# -------------------------
labels = df[["customerID", "Churn"]].copy()
labels["Churn"] = labels["Churn"].map({"Yes": 1, "No": 0}).astype(int)

labels.to_parquet(f"{SILVER_DIR}/churn_labels.parquet", index=False)

# Simple quality checks
assert customers["customerID"].isna().sum() == 0, "Null customerID in customers"
assert customers["customerID"].nunique() == len(customers), "Duplicate customerID in customers"
assert labels["Churn"].isin([0, 1]).all(), "Churn label must be 0/1"

print("✅ Silver tables created:")
print(" - data/silver/customers.parquet")
print(" - data/silver/services.parquet")
print(" - data/silver/billing.parquet")
print(" - data/silver/churn_labels.parquet")
