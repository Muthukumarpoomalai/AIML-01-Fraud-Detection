
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Financial Fraud Detection API")

MODEL_PATH = "models/xgboost/fraud_model_sample.pkl"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "TransactionAmt",
    "card1",
    "addr1",
    "addr2",
    "transactions_last_1h",
    "transactions_last_24h",
    "amount_vs_card_average",
    "device_changed",
    "addr1_changed",
    "addr2_changed"
]

class Transaction(BaseModel):
    TransactionID: int
    TransactionAmt: float
    card1: float | None = None
    addr1: float | None = None
    addr2: float | None = None
    transactions_last_1h: float = 0
    transactions_last_24h: float = 0
    amount_vs_card_average: float = 0
    device_changed: float = 0
    addr1_changed: float = 0
    addr2_changed: float = 0

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):

    try:
        data = transaction.model_dump()

        transaction_id = data.pop("TransactionID")

        input_df = pd.DataFrame(
            [[data.get(feature) for feature in FEATURES]],
            columns=FEATURES
        )

        input_df = input_df.fillna(-1)

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        return {
            "TransactionID": transaction_id,
            "fraud_probability": round(probability, 4),
            "prediction": (
                "Fraud" if probability >= 0.5
                else "Legitimate"
            )
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
