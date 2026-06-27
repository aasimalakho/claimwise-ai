"""
ClaimWise - Synthetic Insurance Claims Data Generator
Generates realistic sample claims data for training the risk/fraud scoring model.
"""
import numpy as np
import pandas as pd
import random
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

random.seed(42)
np.random.seed(42)

CLAIM_TYPES = ["auto", "health", "property", "travel"]
N = 5000

DESCRIPTION_TEMPLATES = {
    "auto": [
        "My car was hit while parked outside my house, bumper and headlight damaged.",
        "Collision on the highway, front end damage and airbag deployed.",
        "Windshield cracked by a falling branch during a storm.",
        "Vehicle stolen from the parking lot overnight.",
        "Side mirror and door dented in a parking lot accident.",
    ],
    "health": [
        "Hospitalized for three days due to severe chest pain, requesting reimbursement for treatment.",
        "Underwent emergency surgery after an accident, submitting hospital bills.",
        "Prescribed medication for a chronic condition, claiming pharmacy expenses.",
        "Diagnostic tests and consultation fees for ongoing treatment.",
        "Physiotherapy sessions following a fracture, claiming session costs.",
    ],
    "property": [
        "Kitchen fire caused smoke damage to walls and ceiling.",
        "Pipe burst in the basement causing water damage to furniture.",
        "Roof damaged during a storm, water leaking into the attic.",
        "Break-in resulted in stolen electronics and damaged front door.",
        "Tree fell on the garage during high winds, structural damage reported.",
    ],
    "travel": [
        "Flight cancelled last minute, requesting reimbursement for missed connection costs.",
        "Luggage lost by the airline during an international transfer.",
        "Trip cut short due to a medical emergency abroad, claiming unused bookings.",
        "Personal belongings stolen from hotel room during vacation.",
        "Delayed flight caused missed prepaid tour, requesting compensation.",
    ],
}



def generate_claims(n=N):
    rows = []
    for i in range(n):
        claim_type = random.choice(CLAIM_TYPES)

        # Base claim amount depends on type
        base_amounts = {"auto": 3500, "health": 2200, "property": 6000, "travel": 900}
        claim_amount = max(50, np.random.normal(base_amounts[claim_type], base_amounts[claim_type] * 0.5))

        policy_age_months = np.random.randint(1, 120)  # how long they've held the policy
        prior_claims_count = np.random.poisson(0.6)
        days_since_policy_start_to_incident = np.random.randint(1, 1000)
        claim_filed_delay_days = np.random.randint(0, 60)  # delay between incident and filing
        documentation_completeness = np.clip(np.random.normal(0.85, 0.15), 0, 1)  # % of required docs provided
        claimant_age = np.random.randint(18, 75)

        # Additional realistic features
        prior_payout_total = round(max(0, np.random.normal(prior_claims_count * 1800, 500)), 2)
        num_documents_submitted = max(0, int(np.random.normal(6 * documentation_completeness, 1.5)))
        claim_to_policy_value_ratio = round(np.clip(np.random.normal(0.35, 0.2), 0.01, 3), 3)
        has_witness = np.random.choice([0, 1], p=[0.4, 0.6])
        channel = random.choice(["online", "agent", "call_center"])

        # --- Fraud risk logic (used to LABEL synthetic data realistically) ---
        risk_score = 0
        if policy_age_months < 3:
            risk_score += 0.3  # new policy, quick claim = suspicious
        if claim_filed_delay_days > 30:
            risk_score += 0.15
        if documentation_completeness < 0.5:
            risk_score += 0.25
        if prior_claims_count >= 3:
            risk_score += 0.2
        if claim_amount > base_amounts[claim_type] * 2.5:
            risk_score += 0.25
        if days_since_policy_start_to_incident < 14:
            risk_score += 0.2
        if claim_to_policy_value_ratio > 0.8:
            risk_score += 0.15
        if has_witness == 0:
            risk_score += 0.1
        if num_documents_submitted < 2:
            risk_score += 0.15
        if channel == "online" and claim_amount > base_amounts[claim_type] * 2:
            risk_score += 0.1

        risk_score += np.random.normal(0, 0.12)  # noise
        is_fraud = 1 if risk_score > 0.30 else 0

        rows.append({
            "claim_id": f"CLM{1000+i}",
            "claim_type": claim_type,
            "description": random.choice(DESCRIPTION_TEMPLATES[claim_type]),
            "claim_amount": round(claim_amount, 2),
            "policy_age_months": policy_age_months,
            "prior_claims_count": prior_claims_count,
            "prior_payout_total": prior_payout_total,
            "days_since_policy_start_to_incident": days_since_policy_start_to_incident,
            "claim_filed_delay_days": claim_filed_delay_days,
            "documentation_completeness": round(documentation_completeness, 2),
            "num_documents_submitted": num_documents_submitted,
            "claim_to_policy_value_ratio": claim_to_policy_value_ratio,
            "has_witness": has_witness,
            "channel": channel,
            "claimant_age": claimant_age,
            "is_fraud": is_fraud,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_claims()
    df.to_csv(os.path.join(BASE_DIR, "data", "claims_data.csv"), index=False)
    print(f"Generated {len(df)} claims")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(df.head())
