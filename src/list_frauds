# -------------------- UPI FRAUD DETECTION - LIST FRAUD TRANSACTIONS --------------------
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv, to_hetero_with_bases
import os

# -------------------- MODEL DEFINITIONS --------------------
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

# -------------------- LOAD MODEL & MAPPINGS --------------------
print("📂 Loading trained model and mappings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    checkpoint = torch.load("model/graph_with_feats.pt", map_location=device, weights_only=False)
    model = EdgeClassifier(checkpoint["metadata"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    data = checkpoint.get("hetero_data")
    if data is None:
        raise ValueError("hetero_data missing in checkpoint.")
    data = data.to(device)
    print("✅ Model and graph data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or graph data: {e}")
    exit()

try:
    mappings = torch.load("model/mappings.pth", weights_only=False)
    u_map = mappings.get("user_map", {})
    m_map = mappings.get("merchant_map", {})
    d_map = mappings.get("device_map", {})
    print("✅ Mappings loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: could not load mappings ({e})")
    u_map, m_map, d_map = {}, {}, {}

# -------------------- LOAD TRANSACTIONS --------------------
try:
    df = pd.read_csv("data/transactions.csv")
except FileNotFoundError:
    print("❌ data/transactions.csv not found.")
    exit()

print(f"📊 Loaded {len(df)} transactions for evaluation.")

# -------------------- INFERENCE --------------------
with torch.no_grad():
    logits = model(data)
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Fraud probability

# -------------------- ATTACH PREDICTIONS --------------------
# Get all edge indices
u_idx, m_idx = data["user", "pays", "merchant"].edge_index.cpu().numpy()

# Reverse mapping: index → id
rev_u_map = {v: k for k, v in u_map.items()}
rev_m_map = {v: k for k, v in m_map.items()}

records = []
for i in range(len(probs)):
    user_id = rev_u_map.get(u_idx[i], f"user_{u_idx[i]}")
    merchant_id = rev_m_map.get(m_idx[i], f"merchant_{m_idx[i]}")
    prob = float(probs[i])
    is_fraud = "Yes" if prob > 0.5 else "No"

    records.append({
        "edge_id": i,
        "user_id": user_id,
        "merchant_id": merchant_id,
        "fraud_probability": round(prob, 4),
        "is_fraud": is_fraud
    })

fraud_df = pd.DataFrame(records)
fraud_df.sort_values(by="fraud_probability", ascending=False, inplace=True)

# -------------------- SAVE FRAUD RESULTS --------------------
os.makedirs("results", exist_ok=True)

fraud_df.to_csv("results/all_edge_predictions.csv", index=False)
fraud_only = fraud_df[fraud_df["is_fraud"] == "Yes"]
fraud_only.to_csv("results/fraud_edges.csv", index=False)

# -------------------- SUMMARY --------------------
print(f"\n✅ Saved all predictions → results/all_edge_predictions.csv")
print(f"🚨 Found {len(fraud_only)} suspected fraud edges → results/fraud_edges.csv")

if len(fraud_only) > 0:
    print("\n🔍 Top 5 Fraudulent Transactions:")
    print(fraud_only.head().to_string(index=False))

print("✅ Done.")
