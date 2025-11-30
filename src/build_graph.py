# src/build_graph.py
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
import numpy as np
os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/transactions.csv")

# create maps
user_map = {k:i for i,k in enumerate(df['user_id'].unique())}
device_map = {k:i for i,k in enumerate(df['device_id'].unique())}
merchant_map = {k:i for i,k in enumerate(df['merchant_id'].unique())}

data = HeteroData()
data['user'].x = torch.ones((len(user_map), 1))     # placeholder features (later replaced)
data['device'].x = torch.ones((len(device_map), 1))
data['merchant'].x = torch.ones((len(merchant_map), 1))

# edge indices
u_idx = df['user_id'].map(user_map).to_numpy()
d_idx = df['device_id'].map(device_map).to_numpy()
m_idx = df['merchant_id'].map(merchant_map).to_numpy()


edge_index_array = np.array([u_idx, d_idx])  # Combine into single NumPy array
data['user', 'uses', 'device'].edge_index = torch.from_numpy(edge_index_array).long()


edge_index_array = np.array([d_idx, m_idx])
data['device', 'processes', 'merchant'].edge_index = torch.from_numpy(edge_index_array).long()

data['user','pays','merchant'].edge_index = torch.tensor([u_idx, m_idx], dtype=torch.long)

# labels for pays edges (edge-level)
data['user','pays','merchant'].y = torch.tensor(df['is_fraud'].to_numpy(), dtype=torch.long)

torch.save({
    'hetero_data': data,
    'user_map': user_map,
    'device_map': device_map,
    'merchant_map': merchant_map,
    'df': df
}, "data/graph_data.pt")

print("Graph construction done. Saved to data/graph_data.pt")