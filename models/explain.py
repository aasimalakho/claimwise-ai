"""
ClaimWise - SHAP Explainability

Generates a human-readable explanation for WHY a claim received its risk score.
This is critical for the "humans remain accountable for high-impact decisions"
theme: an adjuster reviewing a flagged claim shouldn't just see a number (0.71),
they should see WHY (e.g. "new policy + high claim-to-policy ratio + few documents").

Run: python models/explain.py
Requires: pip install shap
"""
import pandas as pd
import joblib
import shap
import numpy as np

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "risk_model_v2.joblib")
TEST_SET_PATH = os.path.join(BASE_DIR, "data", "test_set.csv")

# Friendly names for features, used to turn SHAP output into plain English
FEATURE_LABELS = {
    "claim_amount": "claim amount",
    "policy_age_months": "policy age",
    "prior_claims_count": "number of prior claims",
    "prior_payout_total": "total prior payouts",
    "days_since_policy_start_to_incident": "time since policy start",
    "claim_filed_delay_days": "delay in filing the claim",
    "documentation_completeness": "documentation completeness",
    "num_documents_submitted": "number of documents submitted",
    "claim_to_policy_value_ratio": "claim-to-policy value ratio",
    "has_witness": "witness present",
    "claimant_age": "claimant age",
    "amount_per_policy_month": "claim amount relative to policy age",
    "delay_to_incident_ratio": "filing delay relative to incident timing",
    "docs_per_claim_amount": "documentation relative to claim size",
}


def load_model_and_data():
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_SET_PATH)
    return pipeline, df


def explain_claim(pipeline, claim_row: pd.DataFrame, top_n=3):
    """
    Returns a plain-English explanation of the top contributing factors
    for a single claim's risk score.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["clf"]

    X_transformed = preprocessor.transform(claim_row)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # shap_values shape: (1, n_features) for a single row
    contributions = list(zip(feature_names, shap_values[0]))
    # Sort by absolute contribution, take the top N that PUSH RISK UP (positive SHAP)
    risk_increasing = sorted(
        [c for c in contributions if c[1] > 0], key=lambda x: -x[1]
    )[:top_n]

    reasons = []
    for fname, value in risk_increasing:
        clean_name = fname.replace("remainder__", "").replace("cat__", "")
        clean_name = FEATURE_LABELS.get(clean_name, clean_name)
        reasons.append(clean_name)

    if not reasons:
        return "No strong risk-increasing factors identified; claim appears clean."

    return "Flagged primarily because of: " + ", ".join(reasons) + "."


if __name__ == "__main__":
    pipeline, df = load_model_and_data()

    feature_cols = [c for c in df.columns if c != "is_fraud"]

    print("=== Sample Claim Explanations ===\n")
    for i in range(5):
        row = df.iloc[[i]][feature_cols]
        risk_score = pipeline.predict_proba(row)[0][1]
        explanation = explain_claim(pipeline, row)
        print(f"Claim {i+1}: risk_score={risk_score:.3f}")
        print(f"  -> {explanation}\n")
