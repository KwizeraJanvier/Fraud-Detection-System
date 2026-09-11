"""
Step 4: Final Model Selection & Packaging for Deployment
Bundles the chosen model + its preprocessing logic + tuned threshold
into a single reusable artifact, plus a predict() function to score
new, raw transactions exactly like the ones that arrive in production.

Run this AFTER model_comparison.py has identified the best model.
"""

import pandas as pd
import numpy as np
import joblib
import json

# ------------------------------------------------------------------
# 1. CHOOSE THE FINAL MODEL
# ------------------------------------------------------------------
# Update these two lines based on what model_comparison.py showed as best.
# Example assumes "XGBoost + Graph Features (tuned threshold)" won.
FINAL_MODEL_PATH = "xgboost_with_graph_features_model.pkl"
FINAL_MODEL_NAME = "XGBoost + Graph Features"

model = joblib.load(FINAL_MODEL_PATH)

with open("chosen_threshold.json") as f:
    threshold = json.load(f)["threshold"]

# The exact column order/set the model was trained on — critical to
# preserve, since tree models expect features in a consistent shape.
X_test_graph = pd.read_csv("X_test_with_graph.csv")
FEATURE_COLUMNS = X_test_graph.columns.tolist()

print(f"Packaging model: {FINAL_MODEL_NAME}")
print(f"Threshold: {threshold:.4f}")
print(f"Expected feature columns ({len(FEATURE_COLUMNS)}):\n{FEATURE_COLUMNS}")

# ------------------------------------------------------------------
# 2. PREPROCESSING FUNCTION — mirrors cleaning + feature engineering
#    steps exactly, so a NEW raw transaction can be scored the same way
#    training data was prepared.
# ------------------------------------------------------------------
def preprocess_transaction(raw_df, graph_degree_lookup=None):
    """
    raw_df: DataFrame with raw columns matching the original schema:
        step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
        nameDest, oldbalanceDest, newbalanceDest
        (isFraud / isFlaggedFraud not required for inference)

    graph_degree_lookup: optional dict with 'in_degree' and 'out_degree'
        dicts (precomputed from historical data) to map nameDest/nameOrig
        to their network degree. If None, degrees default to 0
        (safe fallback for brand-new accounts not seen in training graph).
    """
    df = raw_df.copy()

    # --- basic type safety (mirrors clean_fraud_data.py) ---
    df['type'] = df['type'].astype(str)
    df['nameOrig'] = df['nameOrig'].astype(str)
    df['nameDest'] = df['nameDest'].astype(str)
    for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- engineered features (mirrors clean_fraud_data.py) ---
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['origBalanceZero'] = ((df['oldbalanceOrg'] == 0) &
                              (df['newbalanceOrig'] == 0)).astype(int)
    df['destBalanceZero'] = ((df['oldbalanceDest'] == 0) &
                              (df['newbalanceDest'] == 0)).astype(int)
    df['origIsMerchant'] = df['nameOrig'].str.startswith('M').astype(int)
    df['destIsMerchant'] = df['nameDest'].str.startswith('M').astype(int)

    # --- graph features (mirrors extended_modeling_all_approaches.py) ---
    if graph_degree_lookup is not None:
        df['dest_in_degree'] = df['nameDest'].map(
            graph_degree_lookup.get('in_degree', {})).fillna(0)
        df['orig_out_degree'] = df['nameOrig'].map(
            graph_degree_lookup.get('out_degree', {})).fillna(0)
    else:
        df['dest_in_degree'] = 0
        df['orig_out_degree'] = 0

    # --- one-hot encode type, aligned to training columns ---
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)

    # --- drop non-feature columns if present ---
    drop_cols = [c for c in ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    # --- align to the exact training feature set/order ---
    # any dummy column missing in new data (e.g. a type not present in this
    # batch) is added as all-zero; any unexpected column is dropped.
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df


# ------------------------------------------------------------------
# 3. INFERENCE FUNCTION
# ------------------------------------------------------------------
def predict_fraud(raw_df, graph_degree_lookup=None):
    """
    Takes raw transaction(s) and returns fraud probability + flag
    using the packaged model and tuned threshold.
    """
    X_new = preprocess_transaction(raw_df, graph_degree_lookup)
    proba = model.predict_proba(X_new)[:, 1]
    flag = (proba >= threshold).astype(int)

    return pd.DataFrame({
        "fraud_probability": proba,
        "is_flagged_fraud": flag
    })


# ------------------------------------------------------------------
# 4. SAVE COMPLETE DEPLOYMENT BUNDLE
# ------------------------------------------------------------------
deployment_bundle = {
    "model": model,
    "threshold": threshold,
    "feature_columns": FEATURE_COLUMNS,
    "model_name": FINAL_MODEL_NAME,
}
joblib.dump(deployment_bundle, "fraud_detection_deployment_bundle.pkl")
print("\nSaved: fraud_detection_deployment_bundle.pkl")
print("This single file contains everything needed to score new transactions.")

# ------------------------------------------------------------------
# 5. QUICK TEST — SCORE A SAMPLE TRANSACTION
# ------------------------------------------------------------------
sample_transaction = pd.DataFrame([{
    "step": 1,
    "type": "TRANSFER",
    "amount": 5000.0,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C1666544295",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}])

result = predict_fraud(sample_transaction)
print("\nSample transaction scoring result:")
print(result)

# ------------------------------------------------------------------
# 6. HOW TO LOAD AND USE THIS BUNDLE ELSEWHERE (e.g. a scoring service)
# ------------------------------------------------------------------
print("""
--- Usage in a separate script/service ---
import joblib
bundle = joblib.load("fraud_detection_deployment_bundle.pkl")
model = bundle["model"]
threshold = bundle["threshold"]
feature_columns = bundle["feature_columns"]
# Re-run preprocess_transaction() logic (copy the function) before
# calling model.predict_proba() on the aligned features.
""")
