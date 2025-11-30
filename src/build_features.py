# src/build_features.py
import torch
import pandas as pd
from collections import defaultdict
import numpy as np
import pickle
import torch_geometric.data.hetero_data  # ✅ Required for unpickling

import torch.serialization
torch.serialization.add_safe_globals([torch_geometric.data.hetero_data.HeteroData])

# ✅ Load saved raw graph data
obj = torch.load("data/graph_data.pt", weights_only=False)
data = obj['hetero_data']
df = obj['df']

# ---------------------------
# ✅ Helper aggregation function
# ---------------------------
def agg_amounts(grouped, keys):
    """
    Compute statistical and behavioral features for each user/device/merchant:
    count, mean, std, min, max, sum of transaction amounts,
    + ratio of max to mean, and std/mean to capture variability.
    """
    feats = np.zeros((len(keys), 8), dtype=float)  # 8 features now
    for key, idx in keys.items():
        if key in grouped.groups:
            g = grouped.get_group(key)
            amounts = g['amount'].values
            cnt = len(amounts)
            mean_amt = float(np.mean(amounts))
            std_amt = float(np.std(amounts, ddof=0) if len(amounts) > 1 else 0.0)
            min_amt = float(np.min(amounts))
            max_amt = float(np.max(amounts))
            sum_amt = float(np.sum(amounts))
            max_mean_ratio = max_amt / mean_amt if mean_amt != 0 else 0.0
            std_mean_ratio = std_amt / mean_amt if mean_amt != 0 else 0.0

            feats[idx] = [cnt, mean_amt, std_amt, min_amt, max_amt, sum_amt, max_mean_ratio, std_mean_ratio]
        else:
            feats[idx] = np.zeros(8)
    return feats

# ---------------------------
# ✅ Prepare ID mappings
# ---------------------------
u_keys = {k: i for i, k in enumerate(df['user_id'].unique())}
d_keys = {k: i for i, k in enumerate(df['device_id'].unique())}
m_keys = {k: i for i, k in enumerate(df['merchant_id'].unique())}

# ---------------------------
# ✅ Group data for aggregation
# ---------------------------
g_u = df.groupby('user_id')
g_d = df.groupby('device_id')
g_m = df.groupby('merchant_id')

# ---------------------------
# ✅ Compute features
# ---------------------------
user_feats = agg_amounts(g_u, u_keys)
device_feats = agg_amounts(g_d, d_keys)
merchant_feats = agg_amounts(g_m, m_keys)

# ---------------------------
# ✅ Normalize features
# ---------------------------
def normalize(x):
    x = np.nan_to_num(x)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std

user_feats = normalize(user_feats)
device_feats = normalize(device_feats)
merchant_feats = normalize(merchant_feats)

# ---------------------------
# ✅ Attach features to HeteroData object
# ---------------------------
data['user'].x = torch.tensor(user_feats, dtype=torch.float)
data['device'].x = torch.tensor(device_feats, dtype=torch.float)
data['merchant'].x = torch.tensor(merchant_feats, dtype=torch.float)

# ---------------------------
# ✅ Save enriched graph
# ---------------------------
torch.save({
    'hetero_data': data,
    'user_map': obj['user_map'],
    'device_map': obj['device_map'],
    'merchant_map': obj['merchant_map'],
    'df': df
}, "data/graph_with_feats.pt")

print("✅ Features added and saved successfully to data/graph_with_feats.pt")