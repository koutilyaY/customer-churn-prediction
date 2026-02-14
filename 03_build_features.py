import pandas as pd
import os

SILVER_DIR = "data/silver"
GOLD_DIR = "data/gold"

os.makedirs(GOLD_DIR, exist_ok=True)

customers = pd.read_parquet(f"{SILVER_DIR}/customers.parquet")
services  = pd.read_parquet(f"{SILVER_DIR}/services.parquet")
billing   = pd.read_parquet(f"{SILVER_DIR}/billing.parquet")
labels    = pd.read_parquet(f"{SILVER_DIR}/churn_labels.parquet")

# Merge to one modeling table
df = customers.merge(services, on="customerID", how="inner") \
              .merge(billing, on="customerID", how="inner") \
              .merge(labels, on="customerID", how="inner")

# ---------- Feature Engineering (5-year level basics) ----------
# Spend behavior
df["avg_monthly_spend_est"] = df["TotalCharges"] / (df["tenure"] + 1)

# Tenure buckets (better than raw only)
df["tenure_bucket"] = pd.cut(
    df["tenure"],
    bins=[-1, 6, 12, 24, 48, 72, 999],
    labels=["0-6", "7-12", "13-24", "25-48", "49-72", "72+"]
).astype(str)

# High spender flag
monthly_median = df["MonthlyCharges"].median()
df["high_spender"] = (df["MonthlyCharges"] > monthly_median).astype(int)

# Service richness score (how many add-ons)
addon_cols = [
    "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies"
]
def yes_to_1(x):
    return (x == "Yes").astype(int)

for c in addon_cols:
    df[c + "_flag"] = yes_to_1(df[c])

df["addon_count"] = df[[c + "_flag" for c in addon_cols]].sum(axis=1)

# Internet type flags
df["has_internet"] = (df["InternetService"] != "No").astype(int)
df["fiber_internet"] = (df["InternetService"] == "Fiber optic").astype(int)

# Contract length flags (churn driver)
df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
df["is_one_year"] = (df["Contract"] == "One year").astype(int)
df["is_two_year"] = (df["Contract"] == "Two year").astype(int)

# Payment method risk flags
df["auto_pay"] = df["PaymentMethod"].isin(["Bank transfer (automatic)", "Credit card (automatic)"]).astype(int)
df["is_echeck"] = (df["PaymentMethod"] == "Electronic check").astype(int)

# ---------- Finalize ----------
# Keep raw categorical columns too (we'll OneHotEncode in training)
target = "Churn"

out_path = f"{GOLD_DIR}/churn_features.parquet"
df.to_parquet(out_path, index=False)

print(f"✅ Gold feature table created: {out_path}")
print(f"Rows: {len(df)} | Churn rate: {df[target].mean():.3f}")
