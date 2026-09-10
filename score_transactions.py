"""
Deployment Script 1: Batch Scoring
Scores a CSV file of new, raw transactions using the packaged model bundle
(fraud_detection_deployment_bundle.pkl) and writes results to a new CSV.

Usage:
    python score_transactions.py new_transactions.csv scored_output.csv
"""

import sys
import pandas as pd
import numpy as np
import joblib


def preprocess_transaction(raw_df, feature_columns):
    """Mirrors the exact cleaning + feature engineering used in training."""
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

    # align to exact training feature set/order — missing dummy cols become 0
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def load_bundle(bundle_path="fraud_detection_deployment_bundle.pkl"):
    bundle = joblib.load(bundle_path)
    print(f"Loaded model: {bundle['model_name']}  |  threshold={bundle['threshold']:.4f}")
    return bundle


def score_dataframe(raw_df, bundle):
    model = bundle["model"]
    threshold = bundle["threshold"]
    feature_columns = bundle["feature_columns"]
    scaler = bundle.get("scaler")  # only present for models that need scaling (e.g. Logistic Regression)

    X_new = preprocess_transaction(raw_df, feature_columns)

    if scaler is not None:
        X_new = scaler.transform(X_new)   # critical: same fitted scaler as training

    proba = model.predict_proba(X_new)[:, 1]
    flag = (proba >= threshold).astype(int)

    result = raw_df.copy()
    result["fraud_probability"] = proba
    result["is_flagged_fraud"] = flag
    return result


def main():
    if len(sys.argv) != 3:
        print("Usage: python score_transactions.py <input_csv> <output_csv>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    bundle = load_bundle()
    raw_df = pd.read_csv(input_path)
    print(f"Loaded {len(raw_df)} transactions from {input_path}")

    scored_df = score_dataframe(raw_df, bundle)
    scored_df.to_csv(output_path, index=False)

    n_flagged = scored_df["is_flagged_fraud"].sum()
    print(f"Scored {len(scored_df)} transactions — {n_flagged} flagged as fraud")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
