import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV

# Optional: try XGBoost, fallback to GradientBoosting if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

DATA_PATH = "data/gold/churn_features.parquet"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

# -----------------------------
# 1) Create simulated as_of_date
# -----------------------------
# Tenure is months since start. We'll assume all customers observed on a fixed "snapshot date"
snapshot_date = pd.Timestamp("2026-02-01")
df["as_of_date"] = snapshot_date - pd.to_timedelta(df["tenure"] * 30, unit="D")

# -----------------------------
# 2) Time-based split (senior)
# -----------------------------
df = df.sort_values("as_of_date").reset_index(drop=True)

# Split by date percentiles
d1 = df["as_of_date"].quantile(0.70)
d2 = df["as_of_date"].quantile(0.85)

train_df = df[df["as_of_date"] <= d1]
val_df   = df[(df["as_of_date"] > d1) & (df["as_of_date"] <= d2)]
test_df  = df[df["as_of_date"] > d2]

target = "Churn"

# Drop leakage columns / IDs
drop_cols = ["customerID", "as_of_date"]
X_train, y_train = train_df.drop(columns=drop_cols + [target]), train_df[target]
X_val, y_val     = val_df.drop(columns=drop_cols + [target]), val_df[target]
X_test, y_test   = test_df.drop(columns=drop_cols + [target]), test_df[target]

# Identify categorical/numerical columns
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=False))
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# -----------------------------
# 3) Baseline: Logistic Regression
# -----------------------------
log_reg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

log_reg.fit(X_train, y_train)

val_pred_lr = log_reg.predict_proba(X_val)[:, 1]
lr_roc = roc_auc_score(y_val, val_pred_lr)
lr_pr  = average_precision_score(y_val, val_pred_lr)

# -----------------------------
# 4) Champion: XGBoost (or fallback)
# -----------------------------
if HAS_XGB:
    base_model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=42
    )
else:
    base_model = GradientBoostingClassifier(random_state=42)

champ = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", base_model)
])

champ.fit(X_train, y_train)

# Calibrate probabilities (very “real-world”)
calibrator = CalibratedClassifierCV(champ, method="isotonic", cv=3)
calibrator.fit(X_train, y_train)

val_pred_ch = calibrator.predict_proba(X_val)[:, 1]
ch_roc = roc_auc_score(y_val, val_pred_ch)
ch_pr  = average_precision_score(y_val, val_pred_ch)

# -----------------------------
# 5) Choose threshold using ROI
# -----------------------------
# Business assumptions (adjust later)
offer_cost = 15.0          # dollars per targeted customer
revenue_saved = 200.0      # expected retained value per true positive

prec, rec, thr = precision_recall_curve(y_val, val_pred_ch)
# thr has len = len(prec)-1
thr = np.append(thr, 1.0)

best_thr, best_profit = 0.5, -1e18
for t in thr:
    preds = (val_pred_ch >= t).astype(int)
    tp = int(((preds == 1) & (y_val.values == 1)).sum())
    targeted = int((preds == 1).sum())
    profit = (tp * revenue_saved) - (targeted * offer_cost)
    if profit > best_profit:
        best_profit = profit
        best_thr = float(t)

# -----------------------------
# 6) Final test evaluation
# -----------------------------
test_pred = calibrator.predict_proba(X_test)[:, 1]
test_roc = roc_auc_score(y_test, test_pred)
test_pr  = average_precision_score(y_test, test_pred)

test_cls = (test_pred >= best_thr).astype(int)
report = classification_report(y_test, test_cls, digits=4)

metrics = {
    "baseline_logreg": {"val_roc_auc": float(lr_roc), "val_pr_auc": float(lr_pr)},
    "champion": {"val_roc_auc": float(ch_roc), "val_pr_auc": float(ch_pr)},
    "chosen_threshold": best_thr,
    "estimated_val_profit": float(best_profit),
    "test": {"roc_auc": float(test_roc), "pr_auc": float(test_pr)},
    "model_type": "XGBoost+Calibrated" if HAS_XGB else "GradientBoosting+Calibrated",
    "split_dates": {"train_end": str(d1.date()), "val_end": str(d2.date())}
}

# Save artifacts
joblib.dump(calibrator, f"{MODELS_DIR}/churn_model.joblib")
joblib.dump(preprocess, f"{MODELS_DIR}/preprocess.joblib")

with open(f"{MODELS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(f"{REPORTS_DIR}/classification_report.txt", "w") as f:
    f.write(report)

print("✅ Training complete")
print(json.dumps(metrics, indent=2))
print("\nSaved:")
print(" - models/churn_model.joblib")
print(" - models/preprocess.joblib")
print(" - models/metrics.json")
print(" - reports/classification_report.txt")
