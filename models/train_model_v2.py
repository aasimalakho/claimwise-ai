"""
ClaimWise - Risk/Fraud Scoring Model (v2: XGBoost + Feature Engineering + SHAP)

Upgrades over v1 (Random Forest):
  - XGBoost classifier (generally stronger on tabular data than Random Forest)
  - Engineered features (ratios/interactions that capture fraud patterns better)
  - SHAP explainability: for every prediction, we can show WHY a claim was flagged
    (e.g. "flagged because: new policy + high claim-to-policy ratio + missing docs")
    -> this is what lets a human adjuster trust and act on the AI's decision fast.

Run: python models/train_model_v2.py
Requires: pip install xgboost shap (in addition to requirements.txt)
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "claims_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "risk_model_v2.joblib")

NUMERIC_FEATURES = [
    "claim_amount",
    "policy_age_months",
    "prior_claims_count",
    "prior_payout_total",
    "days_since_policy_start_to_incident",
    "claim_filed_delay_days",
    "documentation_completeness",
    "num_documents_submitted",
    "claim_to_policy_value_ratio",
    "has_witness",
    "claimant_age",
    # engineered features (added below)
    "amount_per_policy_month",
    "delay_to_incident_ratio",
    "docs_per_claim_amount",
]
CATEGORICAL_FEATURES = ["claim_type", "channel"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio/interaction features that often carry strong fraud signal."""
    df = df.copy()
    # Higher amount relative to how long they've held the policy = more suspicious
    df["amount_per_policy_month"] = df["claim_amount"] / (df["policy_age_months"] + 1)
    # Filing delay relative to how soon after policy start the incident happened
    df["delay_to_incident_ratio"] = df["claim_filed_delay_days"] / (
        df["days_since_policy_start_to_incident"] + 1
    )
    # Fewer documents for a larger claim = red flag
    df["docs_per_claim_amount"] = df["num_documents_submitted"] / (df["claim_amount"] / 1000 + 1)
    return df


def build_pipeline():
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ])
    return pipeline


def train():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()

    # Lightweight hyperparameter search (keeps runtime fast for a hackathon timeline)
    param_dist = {
        "clf__max_depth": [3, 4, 5, 6],
        "clf__n_estimators": [150, 300, 400],
        "clf__learning_rate": [0.03, 0.05, 0.1],
    }
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=8, scoring="roc_auc", cv=3, random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    preds = best_pipeline.predict(X_test)
    probs = best_pipeline.predict_proba(X_test)[:, 1]

    print("=== Best Params ===")
    print(search.best_params_)
    print("\n=== Classification Report (XGBoost v2) ===")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")

    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Save test set too, so the SHAP explainability script can reuse the exact same split
    X_test.assign(is_fraud=y_test).to_csv(
        os.path.join(BASE_DIR, "data", "test_set.csv"), index=False
    )


if __name__ == "__main__":
    train()
