
import pandas as pd

file_path = "data/raw/train_transaction.csv"

total_transactions = 0
fraud_transactions = 0

# Read only the target column to reduce memory usage
for chunk in pd.read_csv(
    file_path,
    usecols=["isFraud"],
    chunksize=100000
):
    total_transactions += len(chunk)
    fraud_transactions += int(chunk["isFraud"].sum())

legitimate_transactions = total_transactions - fraud_transactions

fraud_percentage = (
    fraud_transactions / total_transactions
) * 100

print("===== FRAUD DATASET ANALYSIS =====")
print("Total Transactions:", total_transactions)
print("Legitimate Transactions:", legitimate_transactions)
print("Fraud Transactions:", fraud_transactions)
print(f"Fraud Percentage: {fraud_percentage:.4f}%")
print(f"Legitimate Percentage: {100 - fraud_percentage:.4f}%")
print("=================================")


import pandas as pd

file_path = "data/raw/train_transaction.csv"

df = pd.read_csv(
    file_path,
    usecols=[
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card4",
        "card6",
        "isFraud"
    ]
)

print("\n===== DATASET OVERVIEW =====")
print("Shape:", df.shape)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== FRAUD BY PRODUCT =====")
print(df.groupby("ProductCD")["isFraud"].agg(
    ["count", "sum", "mean"]
))

print("\n===== TRANSACTION AMOUNT =====")
print(df.groupby("isFraud")["TransactionAmt"].describe())

print("\n===== FRAUD BY CARD TYPE =====")
print(df.groupby("card4")["isFraud"].agg(
    ["count", "sum", "mean"]
))
