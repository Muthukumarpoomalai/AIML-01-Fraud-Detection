
import os
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.metrics import average_precision_score, roc_auc_score

DATA_PATH = "data/processed/engineered_features.csv"
OUTPUT_PATH = "models/xgboost/fraud_model_sample.pkl"

os.makedirs("models/xgboost", exist_ok=True)

df = pd.read_csv(DATA_PATH, nrows=100_000)

drop_columns = [
    "TransactionID",
    "TransactionDT",
    "event_timestamp",
    "DeviceInfo",
    "isFraud"
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df["isFraud"]

X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(-1)

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

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

print("Training XGBoost on 80,000 transactions...")

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

print("XGBoost PR-AUC:", average_precision_score(y_test, probabilities))
print("XGBoost ROC-AUC:", roc_auc_score(y_test, probabilities))

joblib.dump(model, OUTPUT_PATH)

print("Sample XGBoost model saved:", OUTPUT_PATH)
