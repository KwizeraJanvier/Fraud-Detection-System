"""
Switch Deployed Model to XGBoost + Graph Features (the best-performing model)
Repackages the deployment bundle so app.py / fraud_scoring_api.py /
score_transactions.py all pick it up automatically -- no other code changes needed,
since they read model/scaler/threshold/feature_columns from the bundle.

Run this AFTER wire_graph_features.py has produced
xgboost_with_graph_features_model.pkl and X_test_with_graph.csv
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (classification_report, confusion_matrix,
                              average_precision_score, roc_auc_score)

# ------------------------------------------------------------------
# 1. LOAD THE BEST MODEL (from your comparison table)
# ------------------------------------------------------------------
model = joblib.load("xgboost_with_graph_features_model.pkl")
X_test = pd.read_csv("X_test_with_graph.csv")
y_test = pd.read_csv("y_test_with_graph.csv").squeeze()

FEATURE_COLUMNS = X_test.columns.tolist()

# ------------------------------------------------------------------
# 2. USE DEFAULT THRESHOLD (0.5) -- this model doesn't need tuning,
#    since precision and recall are both already ~0.9976 at 0.5.
# ------------------------------------------------------------------
THRESHOLD = 0.5

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype(int)

print("=== Verifying XGBoost + Graph Features before packaging ===")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("PR-AUC:", average_precision_score(y_test, y_proba))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# ------------------------------------------------------------------
# 3. PACKAGE AS THE DEPLOYMENT BUNDLE
# ------------------------------------------------------------------
# NOTE: no scaler needed -- tree-based models like XGBoost aren't
# sensitive to feature scale, unlike Logistic Regression.
deployment_bundle = {
    "model": model,
    "scaler": None,
    "threshold": THRESHOLD,
    "feature_columns": FEATURE_COLUMNS,
    "model_name": "XGBoost + Graph Features",
}
joblib.dump(deployment_bundle, "fraud_detection_deployment_bundle.pkl")
print("\nSaved: fraud_detection_deployment_bundle.pkl")
print("Model deployed: XGBoost + Graph Features (threshold=0.5)")

# ------------------------------------------------------------------
# 4. QUICK SANITY TEST -- same fraud pattern that fooled Logistic Regression
# ------------------------------------------------------------------
def preprocess_transaction(raw_df, feature_columns):
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

    # graph features -- default to 0 for accounts not in the training graph
    df['dest_in_degree'] = 0
    df['orig_out_degree'] = 0

    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)
    drop_cols = [c for c in ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
                 if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


sample_fraud = pd.DataFrame([{
    "step": 1, "type": "TRANSFER", "amount": 5000.0,
    "nameOrig": "C1231006815", "oldbalanceOrg": 5000.0, "newbalanceOrig": 0.0,
    "nameDest": "C1666544295", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}])

X_sample = preprocess_transaction(sample_fraud, FEATURE_COLUMNS)
proba = model.predict_proba(X_sample)[:, 1][0]
print(f"\nSanity check -- fraud-pattern transaction fraud probability: {proba:.4f}")
print("(Should be high -- this is the same case that scored only 0.0001 "
      "with the recalibrated Logistic Regression)")
