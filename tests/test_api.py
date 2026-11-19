from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Fraud Detection API is running"

def test_prediction_endpoint():
    payload = {
        "TransactionID": 2987095,
        "TransactionAmt": 150.0,
        "card1": 10000,
        "addr1": 100,
        "addr2": 87,
        "transactions_last_1h": 2,
        "transactions_last_24h": 5,
        "amount_vs_card_average": 1.2,
        "device_changed": 0,
        "addr1_changed": 0,
        "addr2_changed": 0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["TransactionID"] == 2987095
    assert 0 <= result["fraud_probability"] <= 1
    assert result["prediction"] in ["Fraud", "Legitimate"]
