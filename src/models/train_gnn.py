
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/processed/engineered_features.csv"
MODEL_DIR = "models/gnn"

os.makedirs(MODEL_DIR, exist_ok=True)

MAX_ROWS = 100_000
EPOCHS = 15
HIDDEN_CHANNELS = 32
LEARNING_RATE = 0.003

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")

print("Loading transaction dataset...")

required_columns = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "DeviceInfo",
    "addr1",
    "addr2",
    "transactions_last_1h",
    "transactions_last_24h",
    "amount_vs_card_average",
    "device_changed",
    "addr1_changed",
    "addr2_changed",
    "isFraud"
]

df = pd.read_csv(
    DATA_PATH,
    usecols=lambda column: column in required_columns,
    nrows=MAX_ROWS
)

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

df = df.sort_values("TransactionDT").reset_index(drop=True)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. NODE FEATURES
# --------------------------------------------------

feature_columns = [
    "TransactionAmt",
    "transactions_last_1h",
    "transactions_last_24h",
    "amount_vs_card_average",
    "device_changed",
    "addr1_changed",
    "addr2_changed"
]

X_df = df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

split_index = int(len(df) * 0.8)

# Fit preprocessing only on training data
train_medians = X_df.iloc[:split_index].median().fillna(0)

X_df = X_df.fillna(train_medians)

train_mean = X_df.iloc[:split_index].mean()
train_std = X_df.iloc[:split_index].std().replace(0, 1)

X = ((X_df - train_mean) / train_std).to_numpy(
    dtype=np.float32
)

y = df["isFraud"].to_numpy(dtype=np.int64)

x = torch.tensor(X, dtype=torch.float32)
labels = torch.tensor(y, dtype=torch.long)

train_mask = torch.zeros(len(df), dtype=torch.bool)
test_mask = torch.zeros(len(df), dtype=torch.bool)

train_mask[:split_index] = True
test_mask[split_index:] = True

print("Training transactions:", split_index)
print("Testing transactions:", len(df) - split_index)

# --------------------------------------------------
# 3. GRAPH CONSTRUCTION
# --------------------------------------------------

print("\nBuilding transaction graph...")

entity_columns = [
    "card1",
    "DeviceInfo",
    "addr1",
    "addr2"
]

source_nodes = []
target_nodes = []

row_indices = np.arange(len(df))

for column in entity_columns:

    values = df[column].fillna("").astype(str)

    codes, _ = pd.factorize(values, sort=False)

    valid = codes >= 0

    valid_rows = row_indices[valid]
    valid_codes = codes[valid]

    order = np.lexsort((valid_rows, valid_codes))

    sorted_rows = valid_rows[order]
    sorted_codes = valid_codes[order]

    same_entity = (
        sorted_codes[1:] == sorted_codes[:-1]
    )

    previous_transactions = sorted_rows[:-1][same_entity]
    current_transactions = sorted_rows[1:][same_entity]

    source_nodes.extend(previous_transactions.tolist())
    target_nodes.extend(current_transactions.tolist())

    print(
        f"{column}: {len(current_transactions)} connections"
    )

# Directed edges: previous transaction -> current transaction
edge_index = torch.tensor(
    [source_nodes, target_nodes],
    dtype=torch.long
)

print("Total graph edges:", edge_index.shape[1])

# --------------------------------------------------
# 4. GRAPH NEURAL NETWORK
# --------------------------------------------------

class FraudGNN(nn.Module):

    def __init__(self, input_channels, hidden_channels):

        super().__init__()

        self.conv1 = SAGEConv(
            input_channels,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels
        )

        self.output = nn.Linear(
            hidden_channels,
            2
        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return self.output(x)


model = FraudGNN(
    input_channels=X.shape[1],
    hidden_channels=HIDDEN_CHANNELS
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Handle class imbalance
train_labels = labels[train_mask]

positive_count = (train_labels == 1).sum().item()
negative_count = (train_labels == 0).sum().item()

if positive_count == 0:
    raise ValueError("No fraud transactions found in training data.")

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32
)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        [1.0, negative_count / positive_count],
        dtype=torch.float32
    )
)

print("\nTraining GNN...")
print("Fraud transactions in training:", positive_count)

# --------------------------------------------------
# 5. MODEL TRAINING
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    outputs = model(x, edge_index)

    loss = criterion(
        outputs[train_mask],
        labels[train_mask]
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {loss.item():.4f}"
    )

# --------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------

print("\nEvaluating GNN...")

model.eval()

with torch.no_grad():

    outputs = model(x, edge_index)

    probabilities = torch.softmax(
        outputs[test_mask],
        dim=1
    )[:, 1].numpy()

y_test = y[split_index:]

predictions = (probabilities >= 0.5).astype(int)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# 7. SAVE MODEL AND PREDICTIONS
# --------------------------------------------------

torch.save(
    model.state_dict(),
    f"{MODEL_DIR}/fraud_gnn.pth"
)

test_results = pd.DataFrame({
    "TransactionID": df["TransactionID"].iloc[split_index:].values,
    "actual_fraud": y_test,
    "gnn_fraud_probability": probabilities
})

test_results.to_csv(
    f"{MODEL_DIR}/gnn_test_predictions.csv",
    index=False
)

metrics = {
    "PR_AUC": float(pr_auc),
    "ROC_AUC": float(roc_auc),
    "training_transactions": int(split_index),
    "testing_transactions": int(len(y_test)),
    "graph_edges": int(edge_index.shape[1]),
    "epochs": EPOCHS
}

with open(f"{MODEL_DIR}/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nGNN training completed!")
print("Model saved:", f"{MODEL_DIR}/fraud_gnn.pth")
print("Predictions saved:", f"{MODEL_DIR}/gnn_test_predictions.csv")
print("Metrics saved:", f"{MODEL_DIR}/metrics.json")
