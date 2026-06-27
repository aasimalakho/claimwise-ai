"""
ClaimWise - Risk/Fraud Scoring Model Trainer
Trains a classifier to predict fraud risk on insurance claims.
Outputs a risk PROBABILITY (0-1), not just a label, since the case flow
needs a score to decide: auto-settle (low) / human review (mid) / auto-flag (high).
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "claims_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "risk_model.joblib")

NUMERIC_FEATURES = [
    "claim_amount",
    "policy_age_months",
    "prior_claims_count",
    "days_since_policy_start_to_incident",
    "claim_filed_delay_days",
    "documentation_completeness",
    "claimant_age",
]
CATEGORICAL_FEATURES = ["claim_type"]


def train():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=150, max_depth=6, random_state=42, class_weight="balanced"
        )),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
