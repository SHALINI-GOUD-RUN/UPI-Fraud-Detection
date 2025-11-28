# -------------------- UPI FRAUD DETECTION - TRAINING SCRIPT (FINAL with RESULTS) --------------------
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero_with_bases
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# -------------------- REPRODUCIBILITY --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------- LOAD GRAPH DATA --------------------
print("📂 Loading graph_with_feats.pt ...")
obj = torch.load("data/graph_with_feats.pt", weights_only=False)
data = obj["hetero_data"]

edge_key = ("user", "pays", "merchant")
if edge_key not in data.edge_types:
    raise KeyError(f"Expected edge type {edge_key} not found. Available: {data.edge_types}")

# -------------------- SPLIT TRAIN/TEST --------------------
y_np = data[edge_key].y.cpu().numpy().squeeze()
indices = np.arange(len(y_np))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, stratify=y_np, random_state=SEED
)

train_mask = torch.zeros(len(y_np), dtype=torch.bool)
test_mask = torch.zeros(len(y_np), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data[edge_key].train_mask = train_mask
data[edge_key].test_mask = test_mask

# -------------------- MODEL DEFINITION --------------------
class GNN(torch.nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), hidden)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeClassifier(torch.nn.Module):
    def __init__(self, metadata, hidden=64):
        super().__init__()
        self.gnn = to_hetero_with_bases(GNN(hidden), metadata=metadata, num_bases=3)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2)
        )

    def forward(self, data):
        h = self.gnn(data.x_dict, data.edge_index_dict)
        u_idx, m_idx = data["user", "pays", "merchant"].edge_index
        z = torch.cat([h["user"][u_idx], h["merchant"][m_idx]], dim=1)
        return self.edge_mlp(z)

# -------------------- TRAINING SETUP --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 Using device:", device)

model = EdgeClassifier(data.metadata(), hidden=64).to(device)
data = data.to(device)

edge_labels = data[edge_key].y.long().squeeze()
train_mask = data[edge_key].train_mask
test_mask = data[edge_key].test_mask

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# compute class weights
try:
    counts = torch.bincount(edge_labels)
    inv_freq = counts.sum() / (counts + 1e-9)
    class_weights = inv_freq / inv_freq.sum() * 2.0
except Exception:
    class_weights = torch.tensor([1.0, 5.0], dtype=torch.float)
class_weights = class_weights.to(device)

# -------------------- TRAIN LOOP --------------------
THRESHOLD = 0.5
EPOCHS = 50
print("🚀 Training started...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    logits = model(data)
    loss = F.cross_entropy(logits[train_mask], edge_labels[train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(logits[test_mask], dim=1)[:, 1].cpu().numpy()
            preds = (probs >= THRESHOLD).astype(int)
            y_true = edge_labels[test_mask].cpu().numpy()

            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | "
                  f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
            print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

train_time = time.time() - start_time

# -------------------- FINAL EVAL --------------------
model.eval()
with torch.no_grad():
    logits_final = model(data)
    probs_final = torch.softmax(logits_final[test_mask], dim=1)[:, 1].cpu().numpy()
    preds_final = (probs_final >= THRESHOLD).astype(int)
    y_true_final = edge_labels[test_mask].cpu().numpy()

acc = accuracy_score(y_true_final, preds_final)
prec = precision_score(y_true_final, preds_final, zero_division=0)
rec = recall_score(y_true_final, preds_final, zero_division=0)
f1 = f1_score(y_true_final, preds_final, zero_division=0)
try:
    auc = roc_auc_score(y_true_final, probs_final)
except ValueError:
    auc = float("nan")

print("\n✅ Final Results:")
print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc if not np.isnan(auc) else 'N/A'}")
print(f"Training Time: {train_time:.2f}s")

# -------------------- SAVE RESULTS --------------------
os.makedirs("results", exist_ok=True)
df_res = pd.DataFrame([{
    "Model": "GNN (Proposed)",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "AUC": auc if not np.isnan(auc) else None,
    "Train_Time": round(train_time, 2)
}])
df_res.to_csv("results/gnn_results.csv", index=False)

# -------------------- SAVE TRAINED MODEL & MAPPINGS --------------------
os.makedirs("model", exist_ok=True)

print("💾 Saving trained model and mappings...")

torch.save({
    "model_state_dict": model.state_dict(),
    "metadata": data.metadata(),
    "hetero_data": data  # ✅ include full graph for inference
}, "model/graph_with_feats.pt")

torch.save({
    "user_map": obj.get("user_map", {}),
    "merchant_map": obj.get("merchant_map", {}),
    "device_map": obj.get("device_map", {})
}, "model/mappings.pth")

print("\n✅ Model and mappings saved successfully!")
print("📂 Files created:")
print("   ├─ model/graph_with_feats.pt")
print("   └─ model/mappings.pth")
print("📊 Results saved to: results/gnn_results.csv")
print("✅ Training complete!")
