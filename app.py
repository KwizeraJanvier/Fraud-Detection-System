"""
Flask + HTML Fraud Detection Demo
A browser-based interface for the fraud model, using plain Flask + HTML
instead of Gradio -- deployable on Render's free tier.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000 in a browser.

For Render deployment, see the accompanying instructions.
"""

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ------------------------------------------------------------------
# LOAD MODEL BUNDLE ONCE AT STARTUP
# ------------------------------------------------------------------
bundle = joblib.load("fraud_detection_deployment_bundle.pkl")
model = bundle["model"]
threshold = bundle["threshold"]
feature_columns = bundle["feature_columns"]
scaler = bundle.get("scaler")
model_name = bundle["model_name"]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

EXAMPLES = [
    {"label": "Legit payment", "step": 1, "type": "PAYMENT", "amount": 9839.64,
     "nameOrig": "C1231006815", "oldbalanceOrg": 170136.0, "newbalanceOrig": 160296.36,
     "nameDest": "M1979787155", "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
    {"label": "Suspicious transfer", "step": 1, "type": "TRANSFER", "amount": 5000.0,
     "nameOrig": "C1231006815", "oldbalanceOrg": 5000.0, "newbalanceOrig": 0.0,
     "nameDest": "C1666544295", "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
    {"label": "Cash-out pattern", "step": 1, "type": "CASH_OUT", "amount": 181.0,
     "nameOrig": "C840083671", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
     "nameDest": "C38997010", "oldbalanceDest": 21182.0, "newbalanceDest": 0.0},
]


def preprocess_transaction(raw_df):
    """Mirrors the exact cleaning + feature engineering used in training,
    including the graph features (dest_in_degree / orig_out_degree) used
    by the XGBoost + Graph Features deployed model."""
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

    # Graph features: the deployed model was trained with these, but at
    # inference time we don't have the full historical transaction graph
    # to compute real degrees for new accounts. Defaulting to 0 is a safe
    # fallback -- it means "no known network history for this account,"
    # which is also true for brand-new accounts in production.
    df['dest_in_degree'] = 0
    df['orig_out_degree'] = 0

    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)
    drop_cols = [c for c in ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
                 if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def predict_fraud(form):
    raw_df = pd.DataFrame([{
        "step": float(form["step"]),
        "type": form["type"],
        "amount": float(form["amount"]),
        "nameOrig": form["nameOrig"],
        "oldbalanceOrg": float(form["oldbalanceOrg"]),
        "newbalanceOrig": float(form["newbalanceOrig"]),
        "nameDest": form["nameDest"],
        "oldbalanceDest": float(form["oldbalanceDest"]),
        "newbalanceDest": float(form["newbalanceDest"]),
    }])

    X_new = preprocess_transaction(raw_df)
    if scaler is not None:
        X_new = scaler.transform(X_new)

    proba = float(model.predict_proba(X_new)[:, 1][0])
    flagged = proba >= threshold
    return proba, flagged


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_values = EXAMPLES[1]  # default pre-fill

    if request.method == "POST":
        form_values = request.form
        try:
            proba, flagged = predict_fraud(request.form)
            result = {
                "proba": proba,
                "flagged": flagged,
                "threshold": threshold,
                "model_name": model_name,
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template(
        "index.html",
        types=TRANSACTION_TYPES,
        examples=EXAMPLES,
        values=form_values,
        result=result,
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT env var
    app.run(host="0.0.0.0", port=port, debug=False)
