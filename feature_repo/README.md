
@'
# Feast Feature Store – Financial Fraud Detection

## Overview

This repository contains the Feast feature store implementation for the Real-Time Financial Fraud Detection Platform.

## Configuration

- Feature store: Feast
- Entity: transaction_id
- Feature View: fraud_transaction_features
- Offline source: Engineered transaction Parquet file
- Online store: SQLite
- Registry: SQLite

## Features

The feature view contains:

- TransactionAmt
- transactions_last_1h
- transactions_last_24h
- amount_vs_card_average
- device_changed
- addr1_changed
- addr2_changed

## Project Files

- `features.py` – Entity, source and feature view definitions.
- `feature_store.yaml` – Feast registry and online/offline store configuration.
- `data/` – Project data is stored in the main project's data directory.

## Running Feast

Run these commands from the `feature_repo` directory.

Apply feature definitions:

```powershell
feast apply
```

Materialize features into the online store:

```powershell
feast materialize-incremental 2030-01-01T00:00:00
```

List feature views:

```powershell
feast feature-views list
```

## Online Feature Retrieval

Example:

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

result = store.get_online_features(
    features=[
        "fraud_transaction_features:TransactionAmt",
        "fraud_transaction_features:transactions_last_1h",
    ],
    entity_rows=[
        {"transaction_id": 2987095}
    ],
).to_dict()

print(result)
```

## Implementation Status

- Feast feature definitions created.
- Feature view successfully applied.
- SQLite online store configured.
- Feature materialization executed.
- Online feature retrieval successfully tested.

## Integration Note

Feast online retrieval has been tested independently. It is not yet integrated into the live FastAPI inference endpoint.
'@ | Set-Content -Encoding UTF8 README.md

Get-Content README.md -TotalCount 12
