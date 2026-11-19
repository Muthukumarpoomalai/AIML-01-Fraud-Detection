
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# File paths
DATA_PATH = "data/processed/engineered_features.csv"
XGB_PATH = "models/xgboost/fraud_model_sample.pkl"
GNN_PATH = "models/gnn/gnn_test_predictions.csv"
OUTPUT_PATH = "models/gnn/blended_predictions.csv"

# Load models and predictions
print("Loading XGBoost model and GNN predictions...")

xgb_model = joblib.load(XGB_PATH)
gnn_results = pd.read_csv(GNN_PATH)

df = pd.read_csv(DATA_PATH)

# Match GNN transactions with original dataset
matched = gnn_results.merge(
    df,
    on="TransactionID",
    how="inner",
    suffixes=("_gnn", "_original")
)

if len(matched) != len(gnn_results):
    raise ValueError("Some GNN transactions could not be matched.")

# Prepare XGBoost features in the same way as training
drop_columns = [
    "TransactionID",
    "TransactionDT",
    "event_timestamp",
    "DeviceInfo",
    "isFraud"
]

X = matched.drop(
    columns=drop_columns + [
        "actual_fraud",
        "gnn_fraud_probability"
    ],
    errors="ignore"
)

X = X.replace([np.inf, -np.inf], np.nan).fillna(-1)

# Preserve original model feature order
expected_features = xgb_model.get_booster().feature_names

if expected_features:
    X = X[expected_features]

xgb_probabilities = xgb_model.predict_proba(X)[:, 1]

# Blend XGBoost and GNN scores
gnn_probabilities = matched["gnn_fraud_probability"].to_numpy()

blended_probabilities = (
    0.7 * xgb_probabilities +
    0.3 * gnn_probabilities
)

y_true = matched["actual_fraud"].to_numpy()

# Evaluation
pr_auc = average_precision_score(
    y_true,
    blended_probabilities
)

roc_auc = roc_auc_score(
    y_true,
    blended_probabilities
)

predictions = (
    blended_probabilities >= 0.5
).astype(int)

print("\nBlended Model Classification Report:")
print(
    classification_report(
        y_true,
        predictions,
        zero_division=0
    )
)

print("Blended PR-AUC:", round(pr_auc, 4))
print("Blended ROC-AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, predictions))

# Save results
results = pd.DataFrame({
    "TransactionID": matched["TransactionID"],
    "actual_fraud": y_true,
    "xgboost_probability": xgb_probabilities,
    "gnn_probability": gnn_probabilities,
    "blended_fraud_probability": blended_probabilities
})

os.makedirs("models/gnn", exist_ok=True)

results.to_csv(OUTPUT_PATH, index=False)

with open("models/gnn/blended_metrics.json", "w") as file:
    json.dump({
        "XGBoost_weight": 0.7,
        "GNN_weight": 0.3,
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc_auc),
        "transactions_evaluated": int(len(results))
    }, file, indent=4)

print("\nBlending completed!")
print("Saved:", OUTPUT_PATH)
