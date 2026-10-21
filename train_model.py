"""Train and evaluate the lightweight TruthLens text-classification baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from domain_version.config import CONFIDENCE_THRESHOLD, RANDOM_STATE
from domain_version.predictor import MODEL_DIR, MODEL_PATH, clean_text

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


def load_dataset() -> tuple[pd.DataFrame, dict]:
    """Read the two ISOT files, normalize text, remove empty and duplicate rows."""
    frames = []
    input_counts = {}
    for filename, label in (("True.csv", 0), ("Fake.csv", 1)):
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset file is missing: {path}. Add both ISOT CSV files under data/."
            )
        frame = pd.read_csv(path)
        missing = {"title", "text"} - set(frame.columns)
        if missing:
            raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
        combined = (
            frame["title"].fillna("").astype(str).str.strip()
            + " "
            + frame["text"].fillna("").astype(str).str.strip()
        ).map(clean_text)
        prepared = pd.DataFrame({"text": combined, "label": label})
        input_counts[filename] = int(len(prepared))
        frames.append(prepared)

    all_rows = pd.concat(frames, ignore_index=True)
    empty_count = int(all_rows["text"].eq("").sum())
    all_rows = all_rows.loc[all_rows["text"].ne("")].copy()

    # Remove all exact texts whose supplied labels disagree instead of choosing one.
    label_counts = all_rows.groupby("text", sort=False)["label"].nunique()
    conflicting_texts = set(label_counts[label_counts > 1].index)
    conflict_count = int(all_rows["text"].isin(conflicting_texts).sum())
    if conflicting_texts:
        all_rows = all_rows.loc[~all_rows["text"].isin(conflicting_texts)].copy()

    before_deduplication = len(all_rows)
    all_rows = all_rows.drop_duplicates(subset="text", keep="first")
    duplicates_removed = int(before_deduplication - len(all_rows))
    all_rows = all_rows.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    if all_rows["label"].nunique() != 2 or len(all_rows) < 100:
        raise ValueError("Both classes need enough usable examples to train the model.")

    summary = {
        "input_rows_by_file": input_counts,
        "empty_rows_removed": empty_count,
        "conflicting_rows_removed": conflict_count,
        "duplicate_rows_removed": duplicates_removed,
        "prepared_examples": int(len(all_rows)),
        "prepared_class_counts": {
            "REAL": int((all_rows["label"] == 0).sum()),
            "FAKE": int((all_rows["label"] == 1).sum()),
        },
    }
    return all_rows, summary


def make_model() -> Pipeline:
    """Build a small, CPU-friendly TF-IDF and Logistic Regression pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.97,
                max_features=50_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1_000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Calculate standard binary metrics with REAL then FAKE label order."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["REAL", "FAKE"],
        "per_class": {
            "REAL": {
                "precision": float(precision_score(labels, predictions, pos_label=0, zero_division=0)),
                "recall": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
                "f1": float(f1_score(labels, predictions, pos_label=0, zero_division=0)),
                "support": int((labels == 0).sum()),
            },
            "FAKE": {
                "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
                "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
                "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
                "support": int((labels == 1).sum()),
            },
        },
    }


def main() -> None:
    rows, data_summary = load_dataset()
    train, test = train_test_split(
        rows,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=rows["label"],
    )

    model = make_model()
    print(
        f"Training TF-IDF + Logistic Regression on {len(train):,} examples "
        f"with {len(test):,} held-out test examples...",
        flush=True,
    )
    model.fit(train["text"], train["label"])

    test_scores = model.predict_proba(test["text"]).max(axis=1)
    test_predictions = model.predict(test["text"])
    test_labels = test["label"].to_numpy(dtype=int)
    base_metrics = calculate_metrics(test_labels, test_predictions)
    confident = test_scores >= CONFIDENCE_THRESHOLD
    base_metrics.update({
        "examples": int(len(test)),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "test_coverage_at_threshold": float(confident.mean()),
        "accuracy_on_confident_test_examples": (
            float(accuracy_score(test_labels[confident], test_predictions[confident]))
            if confident.any()
            else None
        ),
        "uncertain_test_examples": int((~confident).sum()),
    })

    metrics = {
        "project": "TruthLens: Explainable Text-Based Fake News Classification",
        "dataset": "ISOT Fake News Dataset (True.csv and Fake.csv)",
        "dataset_source": (
            "True.csv contains Reuters articles; Fake.csv contains articles from "
            "outlets described by the dataset maintainers as unreliable."
        ),
        "source_label_limitation": (
            "Labels are inherited from dataset file/source membership. They are not "
            "independent fact-checks of each claim, and the two classes have different "
            "publisher distributions."
        ),
        "language": "English",
        "input_features": "cleaned title plus full article body",
        "preprocessing": "lowercase, remove URLs, replace punctuation, collapse whitespace",
        "duplicate_policy": (
            "Remove exact duplicate cleaned title-and-body texts before splitting; "
            "remove conflicting duplicate texts if any are present."
        ),
        "split": "stratified random split: 80% train, 20% held-out test",
        "random_state": RANDOM_STATE,
        "threshold_policy": (
            "UNCERTAIN is shown when the highest Logistic Regression class score is "
            "below the fixed application cutoff. Scores are not calibrated truth probabilities."
        ),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "data_preparation": data_summary,
        "split_examples": {
            "train": int(len(train)),
            "test": int(len(test)),
        },
        "model": {
            "features": "word TF-IDF unigrams and bigrams",
            "maximum_vocabulary": 50_000,
            "classifier": "Logistic Regression",
            "class_weight": "balanced",
        },
        "test_metrics": base_metrics,
        "evaluation_note": (
            "These are held-out results for this cleaned, deduplicated ISOT split. "
            "ISOT's real articles are primarily from Reuters, while its fake articles "
            "come from other outlets; performance may reflect publisher, topic, or "
            "writing-style cues. Do not interpret these numbers as accuracy on arbitrary "
            "current news or as fact verification."
        ),
        "external_test_set": False,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\nHeld-out test results:")
    print(f"  Accuracy:          {base_metrics['accuracy']:.3%}")
    print(f"  Macro precision:   {base_metrics['precision_macro']:.3%}")
    print(f"  Macro recall:      {base_metrics['recall_macro']:.3%}")
    print(f"  Macro F1:          {base_metrics['f1_macro']:.3%}")
    print(f"  Coverage at {CONFIDENCE_THRESHOLD:.0%}:    {base_metrics['test_coverage_at_threshold']:.3%}")
    print(
        "  Accuracy when not uncertain: "
        f"{base_metrics['accuracy_on_confident_test_examples']:.3%}"
        if base_metrics["accuracy_on_confident_test_examples"] is not None
        else "  Accuracy when not uncertain: unavailable"
    )
    print(f"  Confusion matrix [REAL, FAKE]: {base_metrics['confusion_matrix']}")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved report: {METRICS_PATH}")


if __name__ == "__main__":
    main()
