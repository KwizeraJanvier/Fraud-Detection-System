"""
Deployment Script 2: Real-Time Scoring API (Flask)
Exposes the packaged model as an HTTP endpoint for real-time transaction
scoring — e.g. called by a payments system at transaction time.

Run:
    pip install flask
    python fraud_scoring_api.py

Then POST a transaction as JSON to http://localhost:5000/score

Example request body:
{
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C1666544295",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}
"""

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ------------------------------------------------------------------
# LOAD MODEL BUNDLE ONCE AT STARTUP (not per-request — expensive to reload)
# ------------------------------------------------------------------
BUNDLE_PATH = "fraud_detection_deployment_bundle.pkl"
bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
threshold = bundle["threshold"]
feature_columns = bundle["feature_columns"]
scaler = bundle.get("scaler")
model_name = bundle["model_name"]

print(f"Loaded model: {model_name}  |  threshold={threshold:.4f}")

REQUIRED_FIELDS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"
]


def preprocess_transaction(raw_df):
    df = raw_df.copy()
    df['type'] = df['type'].astype(str)
    df['nameOrig'] = df['nameOrig'].astype(str)
    df['nameDest'] = df['nameDest'].astype(str)
    for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['origBalanceZero'] = ((df['oldbalanceOrg'] == 0) &
                              (df['newbalanceOrig'] == 0)).astype(int)
    df['destBalanceZero'] = ((df['oldbalanceDest'] == 0) &
                              (df['newbalanceDest'] == 0)).astype(int)
    df['origIsMerchant'] = df['nameOrig'].str.startswith('M').astype(int)
    df['destIsMerchant'] = df['nameDest'].str.startswith('M').astype(int)

    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)
    drop_cols = [c for c in ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
                 if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": model_name, "threshold": threshold})


@app.route("/score", methods=["POST"])
def score():
    payload = request.get_json(force=True)

    if payload is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    raw_df = pd.DataFrame([payload])

    try:
        X_new = preprocess_transaction(raw_df)
        if scaler is not None:
            X_new = scaler.transform(X_new)
        proba = float(model.predict_proba(X_new)[:, 1][0])
        flagged = bool(proba >= threshold)
    except Exception as e:
        return jsonify({"error": f"Scoring failed: {str(e)}"}), 500

    return jsonify({
        "fraud_probability": proba,
        "is_flagged_fraud": flagged,
        "threshold_used": threshold,
        "model": model_name
    })


@app.route("/score_batch", methods=["POST"])
def score_batch():
    """Accepts a JSON list of transactions and scores all of them."""
    payload = request.get_json(force=True)

    if not isinstance(payload, list):
        return jsonify({"error": "Request body must be a JSON array of transactions"}), 400

    raw_df = pd.DataFrame(payload)
    missing = [f for f in REQUIRED_FIELDS if f not in raw_df.columns]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        X_new = preprocess_transaction(raw_df)
        if scaler is not None:
            X_new = scaler.transform(X_new)
        proba = model.predict_proba(X_new)[:, 1]
        flagged = (proba >= threshold)
    except Exception as e:
        return jsonify({"error": f"Scoring failed: {str(e)}"}), 500

    results = [
        {"fraud_probability": float(p), "is_flagged_fraud": bool(f)}
        for p, f in zip(proba, flagged)
    ]
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
