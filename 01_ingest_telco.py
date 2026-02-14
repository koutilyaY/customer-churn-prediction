import pandas as pd
import os
from datetime import datetime

RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
BRONZE_DIR = "data/bronze"

os.makedirs(BRONZE_DIR, exist_ok=True)

df = pd.read_csv(RAW_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

dt = datetime.today().strftime("%Y-%m-%d")
output_dir = f"{BRONZE_DIR}/dt={dt}"
os.makedirs(output_dir, exist_ok=True)

df.to_parquet(f"{output_dir}/telco_raw.parquet", index=False)

print("Bronze ingestion completed.")

