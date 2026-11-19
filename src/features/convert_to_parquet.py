
import pandas as pd

source = "data/processed/engineered_features.csv"
destination = "data/processed/engineered_features.parquet"

df = pd.read_csv(source)

df["event_timestamp"] = pd.to_datetime(
    df["event_timestamp"]
)

df.to_parquet(destination, index=False)

print("Parquet conversion successful!")
print("Shape:", df.shape)
print("Saved:", destination)
