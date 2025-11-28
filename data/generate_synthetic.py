# data/generate_synthetic.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
os.makedirs("data", exist_ok=True)

np.random.seed(42)

users = [f"U{i}" for i in range(200)]
devices = [f"D{i}" for i in range(100)]
merchants = [f"M{i}" for i in range(100)]
n = 8000

rows = []
start = datetime(2025, 1, 1)

# define fraud rings
fraud_devices = np.random.choice(devices, 5, replace=False)
fraud_merchants = np.random.choice(merchants, 5, replace=False)

for i in range(n):
    ts = start + timedelta(seconds=np.random.randint(0, 30 * 24 * 3600))
    user = np.random.choice(users)
    device = np.random.choice(devices)
    merchant = np.random.choice(merchants)
    amount = np.random.randint(100, 5000)

    # pattern: high fraud for suspicious device–merchant combos
    is_fraud = 0
    if (device in fraud_devices and merchant in fraud_merchants):
        amount = np.random.randint(25000, 120000)
        if np.random.rand() < 0.85:
            is_fraud = 1

    rows.append({
        "txn_id": f"T{i}",
        "user_id": user,
        "device_id": device,
        "merchant_id": merchant,
        "amount": amount,
        "is_fraud": is_fraud,
        "ts": ts.isoformat(),
        "city": np.random.choice(["Delhi", "Mumbai", "Chennai", "Hyderabad"])
    })

df = pd.DataFrame(rows)
print(f"Frauds: {df['is_fraud'].sum()} / {len(df)} ({df['is_fraud'].mean()*100:.2f}%)")
df.to_csv("data/transactions.csv", index=False)
