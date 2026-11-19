
import json
import time
import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)

df = pd.read_csv(
    "data/processed/engineered_features.csv",
    nrows=100
)

print("Sending transactions to Kafka...")

for _, row in df.iterrows():
    transaction = row.astype(object).where(pd.notna(row), None).to_dict()

    producer.send(
        "financial-transactions",
        value=transaction
    )

    print("Sent:", transaction["TransactionID"])
    time.sleep(0.2)

producer.flush()
producer.close()

print("All transactions sent!")
