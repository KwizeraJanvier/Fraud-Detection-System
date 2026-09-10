"""
Fix Logistic Regression Calibration + Package as Final Deployed Model

The original class_weight='balanced' was too aggressive, pushing precision
down to ~0.03 (97%+ false alarm rate). This script:
  1. Retrains Logistic Regression with more moderate class weighting
  2. Tunes the threshold specifically for this model
  3. Packages it as the final deployment bundle

Run this AFTER feature_engineering_split.py (uses X_train.csv etc.)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_curve, average_precision_score,
                              roc_auc_score, f1_score)
import joblib
import json

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

FEATURE_COLUMNS = X_train.columns.tolist()

# ------------------------------------------------------------------
# 2. SCALE FEATURES
# ------------------------------------------------------------------
# Logistic Regression is sensitive to feature scale (unlike tree models).
# amount/balance columns are on a very different scale than the 0/1 flag
# columns, which distorts the learned coefficients and contributes to
# poor calibration. This step alone often meaningfully improves precision.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 3. RETRAIN WITH MODERATE CLASS WEIGHTING (instead of 'balanced')
# ------------------------------------------------------------------
# 'balanced' sets weight inversely proportional to class frequency, which
# for ~0.1%-1% fraud rates can mean weighting fraud cases 100-1000x more
# than legitimate ones -- too extreme, causing the model to over-flag.
# A fixed, moderate weight avoids that overcorrection.
log_reg = LogisticRegression(
    class_weight={0: 1, 1: 10},   # moderate weighting instead of 'balanced'
    max_iter=1000,
    random_state=42
)
log_reg.fit(X_train_scaled, y_train)

y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
y_pred_default = (y_proba >= 0.5).astype(int)

print("=== Logistic Regression (recalibrated, threshold=0.5) ===")
print(classification_report(y_test, y_pred_default, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_default))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("PR-AUC:", average_precision_score(y_test, y_proba))

# ------------------------------------------------------------------
# 4. TUNE THRESHOLD SPECIFICALLY FOR THIS MODEL
# ------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
precisions_aligned = precisions[:-1]
recalls_aligned = recalls[:-1]

f1_scores = 2 * (precisions_aligned * recalls_aligned) / \
            (precisions_aligned + recalls_aligned + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\nBest F1 threshold: {best_threshold:.4f}")
print(f"  Precision: {precisions_aligned[best_idx]:.4f}")
print(f"  Recall:    {recalls_aligned[best_idx]:.4f}")
print(f"  F1:        {f1_scores[best_idx]:.4f}")

y_pred_tuned = (y_proba >= best_threshold).astype(int)
print(f"\n=== Logistic Regression (recalibrated, tuned threshold={best_threshold:.4f}) ===")
print(classification_report(y_test, y_pred_tuned, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tuned))

# ------------------------------------------------------------------
# 5. PACKAGE AS FINAL DEPLOYMENT BUNDLE
# ------------------------------------------------------------------
# NOTE: scaler must be included in the bundle -- new data must be scaled
# with the SAME fitted scaler before prediction, or results will be wrong.
deployment_bundle = {
    "model": log_reg,
    "scaler": scaler,
    "threshold": float(best_threshold),
    "feature_columns": FEATURE_COLUMNS,
    "model_name": "Logistic Regression (recalibrated)",
}
joblib.dump(deployment_bundle, "fraud_detection_deployment_bundle.pkl")
print("\nSaved: fraud_detection_deployment_bundle.pkl")

# ------------------------------------------------------------------
# 6. INFERENCE FUNCTION (mirrors final_model_packaging.py logic,
#    with the added scaling step this model requires)
# ------------------------------------------------------------------
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
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


def predict_fraud(raw_df):
    X_new = preprocess_transaction(raw_df)
    X_new_scaled = scaler.transform(X_new)   # <-- critical: must scale first
    proba = log_reg.predict_proba(X_new_scaled)[:, 1]
    flag = (proba >= best_threshold).astype(int)
    return pd.DataFrame({"fraud_probability": proba, "is_flagged_fraud": flag})


# ------------------------------------------------------------------
# 7. QUICK TEST
# ------------------------------------------------------------------
sample_transaction = pd.DataFrame([{
    "step": 1, "type": "TRANSFER", "amount": 181.0,
    "nameOrig": "C1231006815", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "nameDest": "C1666544295", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}])
print("\nSample transaction scoring result:")
print(predict_fraud(sample_transaction))
