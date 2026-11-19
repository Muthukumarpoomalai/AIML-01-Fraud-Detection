
import os
import json
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    classification_report,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/engineered_features.csv"
MODEL_DIR = "models/xgboost"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading engineered dataset...")
df = pd.read_csv(DATA_PATH)

# Remove columns not used as model features
drop_columns = [
    "TransactionID",
    "TransactionDT",
    "event_timestamp",
    "DeviceInfo",
    "isFraud"
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df["isFraud"]

# Handle missing and infinite values
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(-1)

# Chronological 80/20 split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Handle class imbalance
scale_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=2,
    tree_method="hist"
)

mlflow.set_experiment("Financial_Fraud_Detection")

print("Training XGBoost model...")

with mlflow.start_run():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n===== MODEL EVALUATION =====")
    print(classification_report(y_test, predictions))
    print("PR-AUC:", pr_auc)
    print("ROC-AUC:", roc_auc)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    mlflow.log_params({
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_weight
    })

    mlflow.log_metrics({
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    })

    mlflow.xgboost.log_model(model, "xgboost_model")

# Save model and metrics
joblib.dump(model, f"{MODEL_DIR}/fraud_model.pkl")

with open(f"{MODEL_DIR}/metrics.json", "w") as file:
    json.dump({
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }, file, indent=4)

print("\nModel and metrics saved successfully!")
