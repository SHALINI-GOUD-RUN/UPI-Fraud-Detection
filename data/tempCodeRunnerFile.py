# data/generate_synthetic.py
import csv, random, datetime
from pathlib import Path

Path("data").mkdir(exist_ok=True)
OUT = "data/transactions.csv"

# Entities
users = [f"U{str(i).zfill(3)}" for i in range(1, 301)]
devices = [f"D{str(i).zfill(3)}" for i in range(1, 151)]
merchants = [f"M{str(i).zfill(3)}" for i in range(1, 151)]
cities = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Pune"]

# --- Fraud patterns ---
fraud_devices = random.sample(devices, 10)
fraud_merchants = random.sample(merchants, 10)
fraud_users = random.sample(users, 20)

rows = []
txn_id = 1
start = datetime.datetime(2025, 7, 1)

for _ in range(10000):  # 10k transactions
    ts = start + datetime.timedelta(seconds=random.randint(0, 60*24*3600))
    user = random.choice(users)
    device = random.choice(devices)
    merchant = random.choice(merchants)
    city = random.choice(cities)

    # --- Strongly separate fraudulent transactions ---
    # Fraud pattern: suspicious user + suspicious device + suspicious merchant
    if (user in fraud_users and device in fraud_devices) or (device in fraud_devices and merchant in fraud_merchants):
        amount = random.randint(50000, 200000)
        # 95%+ chance of fraud
        is_fraud = 1 if random.random() < 0.97 else 0
    else:
        # Normal pattern: low or mid amount, very low fraud chance
        if random.random() < 0.01:  # only 1% of normal become frauds
            amount = random.randint(20000, 70000)
            is_fraud = 1
        else:
            amount = random.randint(50, 10000)
            is_fraud = 0

    # Add more feature variation (useful for GNN features)
    row = {
        "txn_id": f"T{txn_id}",
        "user_id": user,
        "device_id": device,
        "merchant_id": merchant,
        "amount": amount,
        "ts": ts.isoformat(),
        "city": city,
        "is_fraud": is_fraud
    }
    rows.append(row)
    txn_id += 1

# Write CSV
with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

fraud_count = sum(r["is_fraud"] for r in rows)
print(f"✅ Generated {len(rows)} transactions ({fraud_count} frauds, {fraud_count/len(rows)*100:.2f}% fraud rate)")
print(f"📁 Saved -> {OUT}")
