# ClaimWise AI

### AI-Powered Insurance Claims Investigation & Settlement

ClaimWise AI is an intelligent insurance claims analysis platform that helps insurers process claims faster, detect potential fraud earlier, and provide transparent AI-assisted decision support.

Built for **UiPath AgentHack 2026**, ClaimWise AI integrates with **UiPath Maestro Case** to automate the investigation stage of the claims lifecycle while ensuring every high-impact decision remains explainable and reviewable by human investigators.

---

# The Problem

Insurance companies process thousands of claims every day.

Traditional claim investigation is often:

- Time-consuming
- Resource intensive
- Inconsistent
- Vulnerable to fraudulent submissions

Manual review of every claim slows settlement for genuine customers while increasing operational costs.

---

# Our Solution

ClaimWise AI combines Natural Language Processing, Machine Learning, and Explainable AI to automatically evaluate insurance claims.

For every submitted claim, the platform:

- Understands the claim description
- Predicts the claim category
- Calculates fraud probability
- Explains why the prediction was made
- Recommends the appropriate investigation workflow

Rather than replacing human investigators, ClaimWise AI prioritizes their attention where it matters most.

---

# Key Capabilities

### Intelligent Claim Understanding

Automatically classifies free-text insurance claims into their corresponding claim categories using **TF-IDF + Logistic Regression**.

### Fraud Risk Assessment

Evaluates structured claim information and predicts fraud probability using an **XGBoost** model trained on engineered insurance features.

### Explainable AI

Every prediction includes human-readable explanations generated using **SHAP**, allowing investigators to understand exactly why a claim received its fraud score.

### Automated Case Routing

Based on the predicted fraud probability, ClaimWise AI recommends one of three actions:

- **Auto Settlement**
- **Human Review**
- **Flag for Investigation**

---

# Workflow

```text
Customer submits claim
          │
          ▼
Claim information extracted
          │
          ▼
ClaimWise AI Analysis
     • Claim Classification
     • Fraud Risk Prediction
     • SHAP Explainability
          │
          ▼
Decision Engine
          │
          ├── Auto Settlement
          ├── Human Review
          └── Investigation
```

---

# Integration with UiPath Maestro Case

ClaimWise AI acts as the intelligence layer within the Investigation phase of UiPath Maestro Case.

| UiPath Stage | ClaimWise AI Contribution |
|--------------|--------------------------|
| Intake | Receives extracted claim information from UiPath Document Understanding |
| Investigation | Performs NLP classification, fraud prediction, and explainability |
| Decision | Returns routing recommendation |
| Human Review | Provides SHAP explanations for investigators |
| Settlement | Enables faster approval of low-risk claims |

---

# AI Architecture

The platform combines multiple AI components.

| Component | Purpose |
|-----------|---------|
| TF-IDF + Logistic Regression | Claim Classification |
| XGBoost | Fraud Risk Prediction |
| SHAP | Explainable AI |
| Feature Engineering | Insurance-specific fraud indicators |
| FastAPI | REST API Service |

---

# Project Structure

```text
claimwise/
├── data/
│   ├── generate_data.py        # Synthetic claims dataset generator (5,000 rows)
│   └── claims_data.csv
│
├── models/
│   ├── train_model_v2.py        # XGBoost + feature engineering + hyperparameter search
│   ├── train_nlp_classifier.py  # NLP claim-type classifier
│   ├── explain.py               # SHAP explainability
│   ├── risk_model_v2.joblib     # Trained fraud model
│   └── claim_type_classifier.joblib
│
├── app/
│   └── main.py                  # FastAPI service exposing /process_claim
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md

```

---

# API

### POST `/process_claim`

Processes an insurance claim and returns:

- Claim classification
- Fraud risk score
- Decision recommendation
- Explainable AI insights

### Example Request

```json
{
  "description": "My car was hit while parked outside my house, bumper damaged.",
  "channel": "online",
  "claim_amount": 15000,
  "policy_age_months": 1,
  "prior_claims_count": 4,
  "prior_payout_total": 7000,
  "days_since_policy_start_to_incident": 5,
  "claim_filed_delay_days": 45,
  "documentation_completeness": 0.3,
  "num_documents_submitted": 1,
  "claim_to_policy_value_ratio": 0.9,
  "has_witness": 0,
  "claimant_age": 40
}
```

### Example Response

```json
{
  "claim_type": "auto",
  "risk_score": 0.99,
  "decision": "flag_for_investigation",
  "explanation": "High fraud risk detected. Flagged for investigation before any payout.",
  "risk_factors": [
    "High claim amount relative to policy age",
    "Delayed claim submission",
    "Limited supporting documentation"
  ]
}
```

---

# Technology Stack

- Python
- FastAPI
- XGBoost
- SHAP
- Scikit-learn
- Pandas
- NumPy
- Joblib
- UiPath Maestro Case

---

# Getting Started

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Generate Dataset *(Optional)*

```bash
python data/generate_data.py
```

## Train the Models

```bash
python models/train_model_v2.py
python models/train_nlp_classifier.py
```

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

---

# Deployment

ClaimWise AI can be deployed on:

- Railway
- Hugging Face Spaces
- Any platform supporting FastAPI

Once deployed, the public `/process_claim` endpoint can be integrated directly into the **UiPath Maestro Case** Investigation stage.

---

# Why ClaimWise AI?

- Accelerates insurance claim processing
- Detects potentially fraudulent claims earlier
- Reduces manual investigation workload
- Provides transparent AI recommendations
- Supports human-in-the-loop decision making
- Integrates seamlessly with UiPath Maestro Case
- Designed for enterprise insurance automation

---

# License

This project is licensed under the **MIT License**.

