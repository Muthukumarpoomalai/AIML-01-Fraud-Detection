
import os
import numpy as np
import pandas as pd

# Dataset paths
TRANSACTION_PATH = "data/raw/train_transaction.csv"
IDENTITY_PATH = "data/raw/train_identity.csv"
OUTPUT_PATH = "data/processed/engineered_features.csv"

os.makedirs("data/processed", exist_ok=True)

print("Loading transaction data...")

transactions = pd.read_csv(
    TRANSACTION_PATH,
    usecols=[
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "addr1",
        "addr2",
        "isFraud"
    ]
)

print("Loading identity data...")

identity = pd.read_csv(
    IDENTITY_PATH,
    usecols=[
        "TransactionID",
        "DeviceInfo"
    ]
)

# Merge transaction and identity information
df = transactions.merge(
    identity,
    on="TransactionID",
    how="left"
)

# Sort chronologically
df = df.sort_values("TransactionDT").reset_index(drop=True)

print("Creating features...")

# 1. Transaction velocity
# Previous transactions by card1 within 1 hour and 24 hours

df["transactions_last_1h"] = 0
df["transactions_last_24h"] = 0

for _, group in df.groupby("card1", sort=False):
    timestamps = group["TransactionDT"].to_numpy()

    count_1h = np.arange(len(timestamps)) - np.searchsorted(
        timestamps, timestamps - 3600, side="left"
    )

    count_24h = np.arange(len(timestamps)) - np.searchsorted(
        timestamps, timestamps - 86400, side="left"
    )

    df.loc[group.index, "transactions_last_1h"] = count_1h
    df.loc[group.index, "transactions_last_24h"] = count_24h

# 2. Transaction amount compared with previous card average
previous_average = (
    df.groupby("card1")["TransactionAmt"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

df["amount_vs_card_average"] = (
    df["TransactionAmt"] / previous_average.replace(0, np.nan)
)

# 3. Device changes
df["previous_device"] = (
    df.groupby("card1")["DeviceInfo"].shift(1)
)

df["device_changed"] = (
    df["DeviceInfo"].notna()
    & df["previous_device"].notna()
    & (df["DeviceInfo"] != df["previous_device"])
).astype(int)

# 4. Location changes
for column in ["addr1", "addr2"]:
    previous_location = df.groupby("card1")[column].shift(1)

    df[f"{column}_changed"] = (
        df[column].notna()
        & previous_location.notna()
        & (df[column] != previous_location)
    ).astype(int)

# Remove temporary column
df.drop(columns=["previous_device"], inplace=True)

# Save engineered features
df.to_csv(OUTPUT_PATH, index=False)

print("\n===== FEATURE ENGINEERING COMPLETE =====")
print("Dataset shape:", df.shape)
print("\nCreated features:")
print([
    "transactions_last_1h",
    "transactions_last_24h",
    "amount_vs_card_average",
    "device_changed",
    "addr1_changed",
    "addr2_changed"
])
print("\nSaved to:", OUTPUT_PATH)
df["event_timestamp"] = pd.to_datetime(
    df["TransactionDT"], unit="s",
    origin=pd.Timestamp("2017-11-30")
)