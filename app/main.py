"""
ClaimWise - FastAPI Service
This is the "brain" UiPath Maestro Case calls into for each claim.

Endpoints:
  POST /classify        -> NLP: detect claim type from description
  POST /score_risk       -> ML: fraud/risk probability from claim features
  POST /process_claim    -> Full pipeline: classify + score + ROUTING DECISION

Routing decision logic (what Maestro Case uses to branch the case flow):
  Every claim requires human sign-off -- the AI only recommends, never decides:
  risk_score < 0.3                -> "recommend_settle"        (fast-track human sign-off)
  0.3 <= risk_score < 0.65        -> "human_review"            (full adjuster review)
  risk_score >= 0.65              -> "flag_for_investigation"  (escalated human review)

Run locally:    uvicorn app.main:app --reload --port 8000
Deploy:         Railway / Hugging Face Spaces (same as other projects)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RISK_MODEL_PATH = os.path.join(BASE_DIR, "models", "risk_model_v2.joblib")
NLP_MODEL_PATH = os.path.join(BASE_DIR, "models", "claim_type_classifier.joblib")

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

app = FastAPI(
    title="ClaimWise AI Agent",
    description="Claim classification, fraud risk scoring, and routing decisions for UiPath Maestro Case",
    version="1.0.0",
)

# Load models once at startup
risk_model = joblib.load(RISK_MODEL_PATH)
nlp_model = joblib.load(NLP_MODEL_PATH)


# ---------- Request/Response Schemas ----------

class ClassifyRequest(BaseModel):
    description: str = Field(..., example="My car was hit while parked outside my house.")


class ClassifyResponse(BaseModel):
    claim_type: str
    confidence: float


class RiskRequest(BaseModel):
    claim_type: str = Field(..., example="auto")
    channel: str = Field(..., example="online")
    claim_amount: float = Field(..., example=4500.0)
    policy_age_months: int = Field(..., example=14)
    prior_claims_count: int = Field(..., example=1)
    prior_payout_total: float = Field(..., example=0.0)
    days_since_policy_start_to_incident: int = Field(..., example=200)
    claim_filed_delay_days: int = Field(..., example=5)
    documentation_completeness: float = Field(..., example=0.9, ge=0, le=1)
    num_documents_submitted: int = Field(..., example=5)
    claim_to_policy_value_ratio: float = Field(..., example=0.3)
    has_witness: int = Field(..., example=1, ge=0, le=1)
    claimant_age: int = Field(..., example=35)


class RiskResponse(BaseModel):
    risk_score: float
    risk_label: str
    explanation: str


class ProcessClaimRequest(BaseModel):
    description: str
    channel: str
    claim_amount: float
    policy_age_months: int
    prior_claims_count: int
    prior_payout_total: float
    days_since_policy_start_to_incident: int
    claim_filed_delay_days: int
    documentation_completeness: float = Field(..., ge=0, le=1)
    num_documents_submitted: int
    claim_to_policy_value_ratio: float
    has_witness: int = Field(..., ge=0, le=1)
    claimant_age: int


class ProcessClaimResponse(BaseModel):
    claim_type: str
    risk_score: float
    decision: str
    explanation: str
    risk_factors: str


# ---------- Helper logic ----------

def _engineer_features(features: dict) -> dict:
    """Add the same ratio/interaction features the model was trained on."""
    features = dict(features)
    features["amount_per_policy_month"] = features["claim_amount"] / (features["policy_age_months"] + 1)
    features["delay_to_incident_ratio"] = features["claim_filed_delay_days"] / (
        features["days_since_policy_start_to_incident"] + 1
    )
    features["docs_per_claim_amount"] = features["num_documents_submitted"] / (
        features["claim_amount"] / 1000 + 1
    )
    return features


def _classify(description: str):
    probs = nlp_model.predict_proba([description])[0]
    classes = nlp_model.classes_
    best_idx = probs.argmax()
    return classes[best_idx], float(probs[best_idx])


def _score_risk(features: dict) -> float:
    enriched = _engineer_features(features)
    df = pd.DataFrame([enriched])
    prob = risk_model.predict_proba(df)[0][1]
    return float(prob)


def _explain_risk(features: dict, top_n=3) -> str:
    """SHAP-based explanation of which factors pushed the risk score up."""
    if not SHAP_AVAILABLE:
        return "Explainability unavailable (install 'shap' to enable)."

    enriched = _engineer_features(features)
    df = pd.DataFrame([enriched])

    preprocessor = risk_model.named_steps["preprocess"]
    model = risk_model.named_steps["clf"]

    X_transformed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    contributions = list(zip(feature_names, shap_values[0]))
    risk_increasing = sorted(
        [c for c in contributions if c[1] > 0], key=lambda x: -x[1]
    )[:top_n]

    if not risk_increasing:
        return "No strong risk-increasing factors identified; claim appears clean."

    reasons = []
    for fname, _ in risk_increasing:
        clean_name = fname.replace("remainder__", "").replace("cat__", "")
        # strip one-hot suffix like claim_type_auto -> claim_type
        for base in FEATURE_LABELS:
            if clean_name.startswith(base):
                clean_name = base
                break
        reasons.append(FEATURE_LABELS.get(clean_name, clean_name))

    return "Flagged primarily because of: " + ", ".join(reasons) + "."


def _decide(risk_score: float) -> tuple[str, str]:
    """
    IMPORTANT: No claim is ever auto-settled without a human checkpoint.
    The AI only RECOMMENDS an action; a human adjuster always makes the
    final settlement decision. This keeps humans accountable for every
    payout, per the principle that humans remain in control of high-impact
    decisions -- the AI's role is to triage and prioritize, not to decide.

    - Low risk    -> "recommend_settle": fast-track for a quick human sign-off
    - Medium risk -> "human_review": full adjuster review required
    - High risk   -> "flag_for_investigation": escalated review, investigation required
    """
    if risk_score < 0.30:
        return (
            "recommend_settle",
            "Low fraud risk and complete documentation. Recommended for fast-track "
            "settlement, pending a human adjuster's final sign-off.",
        )
    elif risk_score < 0.65:
        return (
            "human_review",
            "Risk indicators are ambiguous. Routed to a human adjuster for full review "
            "before any decision is made.",
        )
    else:
        return (
            "flag_for_investigation",
            "High fraud risk detected. Escalated for investigation; a human adjuster "
            "must review before any payout decision.",
        )


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"service": "ClaimWise AI Agent", "status": "running"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    try:
        claim_type, confidence = _classify(req.description)
        return ClassifyResponse(claim_type=claim_type, confidence=round(confidence, 3))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_risk", response_model=RiskResponse)
def score_risk(req: RiskRequest):
    try:
        features = req.dict()
        risk_score = _score_risk(features)
        decision, _ = _decide(risk_score)
        explanation = _explain_risk(features)
        return RiskResponse(
            risk_score=round(risk_score, 3), risk_label=decision, explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_claim", response_model=ProcessClaimResponse)
def process_claim(req: ProcessClaimRequest):
    """
    Full pipeline UiPath Maestro Case calls at the Investigation stage:
    classify claim type -> score risk -> explain -> return routing decision.
    """
    try:
        claim_type, _ = _classify(req.description)

        features = {
            "claim_type": claim_type,
            "channel": req.channel,
            "claim_amount": req.claim_amount,
            "policy_age_months": req.policy_age_months,
            "prior_claims_count": req.prior_claims_count,
            "prior_payout_total": req.prior_payout_total,
            "days_since_policy_start_to_incident": req.days_since_policy_start_to_incident,
            "claim_filed_delay_days": req.claim_filed_delay_days,
            "documentation_completeness": req.documentation_completeness,
            "num_documents_submitted": req.num_documents_submitted,
            "claim_to_policy_value_ratio": req.claim_to_policy_value_ratio,
            "has_witness": req.has_witness,
            "claimant_age": req.claimant_age,
        }
        risk_score = _score_risk(features)
        decision, explanation = _decide(risk_score)
        risk_factors = _explain_risk(features)

        return ProcessClaimResponse(
            claim_type=claim_type,
            risk_score=round(risk_score, 3),
            decision=decision,
            explanation=explanation,
            risk_factors=risk_factors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
