
import json
import math
import requests

from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "financial-transactions",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="fraud-api-consumer",
    value_deserializer=lambda value: json.loads(value.decode("utf-8"))
)

API_URL = "http://127.0.0.1:8000/predict"


def clean_value(value, default=0):
    if value is None:
        return default

    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default


print("Kafka Fraud Detection Consumer started...")

for message in consumer:

    transaction = message.value

    payload = {
        "TransactionID": int(transaction["TransactionID"]),
        "TransactionAmt": clean_value(transaction.get("TransactionAmt")),
        "card1": clean_value(transaction.get("card1"), None),
        "addr1": clean_value(transaction.get("addr1"), None),
        "addr2": clean_value(transaction.get("addr2"), None),
        "transactions_last_1h": clean_value(transaction.get("transactions_last_1h")),
        "transactions_last_24h": clean_value(transaction.get("transactions_last_24h")),
        "amount_vs_card_average": clean_value(transaction.get("amount_vs_card_average")),
        "device_changed": clean_value(transaction.get("device_changed")),
        "addr1_changed": clean_value(transaction.get("addr1_changed")),
        "addr2_changed": clean_value(transaction.get("addr2_changed"))
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        if response.status_code != 200:
            print("Status:", response.status_code)
            print("Validation Error:", response.text)
            continue

        result = response.json()

        print(
            f"Transaction: {result['TransactionID']} | "
            f"Fraud Probability: {result['fraud_probability']} | "
            f"Prediction: {result['prediction']}"
        )

    except requests.RequestException as error:
        print("API request failed:", error)
