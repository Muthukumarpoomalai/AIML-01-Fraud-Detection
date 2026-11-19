
import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

# Dataset paths
DATA_PATH = "data/processed/engineered_features.csv"

OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features to monitor
features = [
    "TransactionAmt",
    "transactions_last_1h",
    "transactions_last_24h",
    "amount_vs_card_average",
    "device_changed",
    "addr1_changed",
    "addr2_changed"
]

print("Loading reference and current data...")

# Read only required columns
df = pd.read_csv(
    DATA_PATH,
    usecols=features,
    nrows=100_000
)

# Reference: first 80,000 transactions
reference_data = df.iloc[:80_000].copy()

# Current: next 20,000 transactions
current_data = df.iloc[80_000:100_000].copy()

# Generate drift report
print("Running Evidently drift detection...")

report = Report([
    DataDriftPreset()
])

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save HTML report
result.save_html(
    f"{OUTPUT_DIR}/data_drift_report.html"
)

print("\nDrift monitoring completed!")
print("Report saved: reports/data_drift_report.html")
