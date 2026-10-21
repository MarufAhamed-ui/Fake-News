"""Standalone model inference and local feature explanations."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from config import CONFIDENCE_THRESHOLD

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
SPACE_RE = re.compile(r"\s+")
LABEL_NAMES = {0: "REAL", 1: "FAKE"}


def clean_text(value: object) -> str:
    """Apply the same deterministic text cleanup during training and prediction."""
    text = str(value or "").lower()
    text = URL_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


@lru_cache(maxsize=2)
def _load_model_for_signature(signature: int):
    """Load one pipeline per saved-model version."""
    del signature
    return joblib.load(MODEL_PATH)


def load_model():
    """Load the current trained pipeline and reuse it until the artifact changes."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train_model.py first."
        )
    return _load_model_for_signature(MODEL_PATH.stat().st_mtime_ns)


def load_metrics() -> dict:
    """Read the model report, returning an empty mapping if it is not present."""
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def analyze_text(value: object, model=None, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """Return a model estimate and the largest local feature contributions."""
    raw_text = str(value or "").strip()
    cleaned = clean_text(raw_text)
    if not cleaned:
        raise ValueError("Enter a headline or article before analyzing it.")

    pipeline = model if model is not None else load_model()
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    features = vectorizer.transform([cleaned])
    if features.nnz == 0:
        return {
            "prediction": "UNCERTAIN",
            "class_prediction": None,
            "score": 0.0,
            "probabilities": {"REAL": 0.0, "FAKE": 0.0},
            "reason": "The text has no terms in the model's learned vocabulary.",
            "fake_terms": [],
            "real_terms": [],
            "recognized_features": 0,
        }

    probabilities = pipeline.predict_proba([cleaned])[0]
    classes = classifier.classes_.astype(int)
    probability_by_label = {
        LABEL_NAMES[int(label)]: float(probability)
        for label, probability in zip(classes, probabilities)
    }
    best_index = int(np.argmax(probabilities))
    class_prediction = LABEL_NAMES[int(classes[best_index])]
    score = float(probabilities[best_index])
    prediction = class_prediction if score >= threshold else "UNCERTAIN"
    reason = (
        "The highest model score is below the configured cutoff."
        if prediction == "UNCERTAIN"
        else "This is a text-classification estimate, not a fact check."
    )

    contributions = features.multiply(classifier.coef_[0]).tocoo()
    names = vectorizer.get_feature_names_out()
    fake_terms = []
    real_terms = []
    for feature_index, contribution in zip(contributions.col, contributions.data):
        item = {"term": str(names[feature_index]), "weight": float(contribution)}
        if contribution > 0:
            fake_terms.append(item)
        elif contribution < 0:
            real_terms.append(item)
    fake_terms.sort(key=lambda item: item["weight"], reverse=True)
    real_terms.sort(key=lambda item: item["weight"])

    return {
        "prediction": prediction,
        "class_prediction": class_prediction,
        "score": score,
        "probabilities": probability_by_label,
        "reason": reason,
        "fake_terms": fake_terms[:5],
        "real_terms": real_terms[:5],
        "recognized_features": int(features.nnz),
    }
