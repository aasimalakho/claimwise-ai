# ClaimWise AI

### AI-Powered Insurance Claims Investigation & Settlement

Built for **UiPath AgentHack 2026**
---

# Project Description

Insurance claims processing is often slow, labor-intensive, and vulnerable to fraud, requiring adjusters to manually review large volumes of claims while ensuring legitimate customers receive timely settlements.

**ClaimWise AI** is an AI-powered insurance claims investigation and case management solution that combines **UiPath Maestro Case**, **API Workflows**, and **machine learning** to streamline the claims lifecycle. It classifies claim types, predicts fraud risk, and provides explainable AI insights to support insurance adjusters during investigations.

Using **UiPath Maestro Case**, every claim progresses through **Intake**, **Investigation**, **Human Review**, and **Settlement**, ensuring that AI augments human decision-making rather than replacing it. The result is faster investigations, improved consistency, enhanced fraud detection, and transparent, human-centered claim processing.

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

# UiPath Components Used

ClaimWise AI leverages the following UiPath capabilities:

- UiPath Maestro Case
- UiPath API Workflows
- UiPath Automation Cloud
- Case Rules

---

# Agent Type

ClaimWise AI combines a **coded AI agent** with UiPath orchestration.

- **Coded Agent:** FastAPI-based machine learning service for claim classification, fraud risk prediction, and explainable AI.
- **UiPath Maestro Case:** Orchestrates the insurance claims lifecycle.
- **UiPath API Workflows:** Connects UiPath with the external AI service.

This solution uses a coded AI agent integrated with UiPath Maestro Case and API Workflows. It does not rely on UiPath Agent Builder.


---

# Key Capabilities

### Intelligent Claim Understanding

Automatically classifies free-text insurance claims into their corresponding claim categories using **TF-IDF + Logistic Regression**.

### Fraud Risk Assessment

Evaluates structured claim information and predicts fraud probability using an **XGBoost** model trained on engineered insurance features.

### Explainable AI

Every prediction includes human-readable explanations generated using **SHAP**, allowing investigators to understand exactly why a claim received its fraud score.

### Automated Case Routing

Based on the AI analysis, ClaimWise AI recommends one of the following actions:

- **Recommend Settlement**
- **Human Review**
- **Flag for Investigation**

---

# Solution Architecture

```text
Customer submits claim
          │
          ▼
UiPath Maestro Case
          │
          ▼
Intake
          │
          ▼
Investigation
          │
          ▼
UiPath API Workflow
          │
          ▼
ClaimWise AI (FastAPI)
          │
    ┌─────┼────────────────────────┐
    │     │                        │
    ▼     ▼                        ▼
Claim  Fraud Risk           SHAP Explainable
Type    Prediction          AI Insights
Classification
    └──────────────┬───────────────┘
                   ▼
           Human Review
        (Adjuster Decision)
                   │
                   ▼
             Settlement
```

---

# Integration with UiPath Maestro Case

ClaimWise AI acts as the intelligence layer within the Investigation phase of UiPath Maestro Case.

| UiPath Stage | ClaimWise AI Contribution |
|--------------|--------------------------|
| Intake | Receives extracted claim information from UiPath Document Understanding |
| Investigation | Performs NLP classification, fraud prediction, and explainability |
| Human Review | Provides SHAP explanations for investigators |
| Settlement | Supports human review with AI-generated recommendations before settlement |

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
- UiPath API Workflows
- UiPath Automation Cloud
- Hugging Face Spaces
- Swagger UI

---

# Repository Contents

- Source code for the FastAPI AI service
- Trained machine learning models
- Model training scripts
- Synthetic dataset generator
- Docker configuration
- API documentation
- Project documentation

---

# Setup Instructions

## Prerequisites

- Python 3.11+
- UiPath Automation Cloud
- UiPath Maestro Case
- UiPath API Workflows

## 1. Clone the Repository

```bash
git clone https://github.com/aasimalakho/claimwise-ai.git
cd <your-repository>
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Generate the Dataset (Optional)

```bash
python data/generate_data.py
```

## 4. Train the Models

```bash
python models/train_model_v2.py
python models/train_nlp_classifier.py
```

## 5. Run the FastAPI Application

```bash
uvicorn app.main:app --reload --port 8000
```

The interactive API documentation will be available at:

```text
http://localhost:8000/docs
```

## 6. Deploy the API

Deploy the FastAPI application to Hugging Face Spaces (used in this project) or any FastAPI-compatible hosting platform. Once deployed, the API endpoint will be available for integration with UiPath.

## 7. Configure UiPath

1. Create an **API Workflow** in UiPath Automation Cloud.
2. Configure the HTTP Request activity to call the deployed `/process_claim` endpoint.
3. Publish the API Workflow.
4. Reference the published API Workflow in the **Investigation** stage of **UiPath Maestro Case**.
5. Configure the required input arguments and map the API response to the workflow outputs.

## 8. Run the Solution

Start a new case in **UiPath Maestro Case**. The Investigation stage invokes the API Workflow, which calls the deployed ClaimWise AI service. The AI returns the claim classification, fraud risk score, explanation, and recommendation for **Human Review** before the case proceeds to **Settlement**.

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

