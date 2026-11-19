
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

transaction = Entity(
    name="transaction_id",
    join_keys=["TransactionID"],
)

fraud_source = FileSource(
    name="fraud_transactions",
    path="../data/processed/engineered_features.parquet",
    timestamp_field="event_timestamp",
)

fraud_features = FeatureView(
    name="fraud_transaction_features",
    entities=[transaction],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="TransactionAmt", dtype=Float64),
        Field(name="transactions_last_1h", dtype=Int64),
        Field(name="transactions_last_24h", dtype=Int64),
        Field(name="amount_vs_card_average", dtype=Float64),
        Field(name="device_changed", dtype=Int64),
        Field(name="addr1_changed", dtype=Int64),
        Field(name="addr2_changed", dtype=Int64),
    ],
    source=fraud_source,
)
