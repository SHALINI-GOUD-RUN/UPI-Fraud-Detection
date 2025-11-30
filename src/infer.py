# -------------------- UPI FRAUD DETECTION - INFERENCE SCRIPT --------------------
import pandas as pd
import os

# -------------------- LOAD FRAUD LIST --------------------
FRAUD_FILE = "results/fraud_edges.csv"

def load_fraud_list():
    """Load fraud_edges.csv to identify fraudulent users or merchants."""
    if not os.path.exists(FRAUD_FILE):
        print("⚠️ fraud_edges.csv not found. No fraud data loaded.")
        return set(), set()

    df = pd.read_csv(FRAUD_FILE)
    fraud_users = set(df["user_id"].astype(str).tolist())
    fraud_merchants = set(df["merchant_id"].astype(str).tolist())

    print(f"✅ Loaded {len(fraud_users)} fraud users and {len(fraud_merchants)} fraud merchants.")
    return fraud_users, fraud_merchants


# Load fraud data once
fraud_users, fraud_merchants = load_fraud_list()


# -------------------- FRAUD CHECK FUNCTION --------------------
def check_transaction(user_id: str, merchant_id: str, device_id: str, amount: float):
    """
    Check whether either user or merchant is flagged as fraudulent.
    Flags as fraud if user or merchant OR BOTH are in fraud list.
    """
    if not fraud_users and not fraud_merchants:
        return {
            "user_id": user_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "amount": amount,
            "probability": 0.05,
            "note": "⚠️ Fraud data not loaded. Defaulting to safe."
        }

    # Normalize IDs
    user_id = str(user_id).strip()
    merchant_id = str(merchant_id).strip()
    device_id = str(device_id).strip()

    # 🚨 Check if either user OR merchant are in fraud lists
    # 🚨 Correct logic
    user_flagged = (user_id in fraud_users) or (user_id in fraud_merchants)
    merchant_flagged = (merchant_id in fraud_merchants) or (merchant_id in fraud_users)

    # 🟥 Both involved in fraud
    if user_flagged and merchant_flagged:
        return {
            "user_id": user_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "amount": amount,
            "probability": 0.995,
            "note": f"🚨 ALERT! Both sender {user_id} and receiver {merchant_id} have prior fraud history."
        }

    # 🟧 Only user flagged
    elif user_flagged:
        return {
            "user_id": user_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "amount": amount,
            "probability": 0.99,
            "note": f"⚠️ Sender {user_id} is linked to fraud history."
        }

    # 🟨 Only merchant flagged
    elif merchant_flagged:
        return {
            "user_id": user_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "amount": amount,
            "probability": 0.98,
            "note": f"⚠️ Receiver (merchant {merchant_id}) is flagged for fraud."
        }

    # ✅ Safe transaction
    else:
        return {
            "user_id": user_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "amount": amount,
            "probability": 0.05,
            "note": "✅ Transaction appears safe."
        }


# -------------------- TEST --------------------
if __name__ == "__main__":
    result = check_transaction("U82", "M54", "D73", 3597)
    print(result)
