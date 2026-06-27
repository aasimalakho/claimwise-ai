"""
ClaimWise - NLP Claim Type Classifier
Classifies claim type (auto / health / property / travel) from the free-text
description submitted with a claim. This simulates the real-world scenario
where an adjuster/agent reads the claim narrative and routes it accordingly.
Uses TF-IDF + Logistic Regression: fast, free, explainable, no external API needed.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "claims_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "claim_type_classifier.joblib")


def train():
    df = pd.read_csv(DATA_PATH)
    X = df["description"]
    y = df["claim_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print("=== NLP Claim Type Classifier Report ===")
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
