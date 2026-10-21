"""Streamlit defense demo for the TruthLens text-classification baseline."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from domain_version.config import CONFIDENCE_THRESHOLD
from domain_version.news import clear_news_cache, fetch_latest_news
from domain_version.predictor import (
    MODEL_PATH,
    analyze_text,
    load_metrics,
    load_model,
)

ROOT = Path(__file__).resolve().parent
HISTORY_PATH = ROOT / "history.csv"
PAGES = ["Home", "Analyze News", "Latest News", "Model Performance", "History", "About"]

st.set_page_config(
    page_title="TruthLens | Explainable News Classifier",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fc; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5eaf1; }
    .hero { padding: 1.6rem 1.8rem; border-radius: 18px;
            background: linear-gradient(120deg,#eaf3ff,#ffffff); border: 1px solid #dae7f5; }
    .hero h1 { margin: 0; color: #17324d; }
    .hero p { color: #526477; font-size: 1.05rem; margin: .4rem 0 0; }
    .result { padding: 1rem 1.2rem; border-radius: 14px; background: white;
              border: 1px solid #dfe7ef; border-left: 6px solid #627d98; }
    .result.real { border-left-color: #27836b; }
    .result.fake { border-left-color: #cb5c55; }
    .result.uncertain { border-left-color: #cf9b2c; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_model():
    """Return a cached model or a clear missing-artifact message."""
    signature = MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else 0
    return _load_cached_model(signature)


@st.cache_resource
def _load_cached_model(signature: int):
    del signature
    return load_model()


def ensure_history() -> None:
    if not HISTORY_PATH.exists():
        with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(["timestamp", "text", "prediction", "score"])


def record_analysis(text: str, result: dict) -> None:
    ensure_history()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            datetime.now().astimezone().isoformat(timespec="seconds"),
            text.replace("\n", " ").strip(),
            result["prediction"],
            f"{result['score']:.6f}",
        ])


def show_result(result: dict) -> None:
    style = result["prediction"].lower()
    st.markdown(
        f'<div class="result {style}"><strong>Model estimate</strong><br>'
        f'<span style="font-size:1.8rem;font-weight:800">{result["prediction"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    left.metric("Highest class score", f'{result["score"]:.1%}')
    right.metric("Recognized text features", f'{result["recognized_features"]:,}')
    st.progress(min(max(result["score"], 0.0), 1.0))
    st.caption(
        "The score is produced by Logistic Regression. It is not a calibrated "
        "probability that the article is true or false."
    )
    if result["reason"]:
        st.info(result["reason"])

    a, b = st.columns(2)
    with a:
        st.markdown("**Terms contributing toward FAKE**")
        if result["fake_terms"]:
            st.write(" · ".join(
                f'{item["term"]} ({item["weight"]:+.3f})'
                for item in result["fake_terms"]
            ))
        else:
            st.caption("No strong positive feature contributions.")
    with b:
        st.markdown("**Terms contributing toward REAL**")
        if result["real_terms"]:
            st.write(" · ".join(
                f'{item["term"]} ({item["weight"]:+.3f})'
                for item in result["real_terms"]
            ))
        else:
            st.caption("No strong negative feature contributions.")
    st.caption(
        "The highlighted terms are local TF-IDF × Logistic Regression "
        "contributions. They explain the model score; they do not prove a claim."
    )


def render_home(metrics: dict, available: bool) -> None:
    st.markdown(
        '<div class="hero"><h1>🔎 TruthLens</h1>'
        '<p>Explainable text-based fake-news classification</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.write(
        "Analyze a news headline or article with a lightweight English text classifier. "
        "The system returns a REAL, FAKE, or UNCERTAIN model estimate and shows the "
        "text features that influenced its score."
    )
    if not available:
        st.error(
            "The trained model is not available yet. Run "
            "python train_model.py from the TruthLens folder."
        )
    st.info(
        "This is an explainable text-only baseline. It does not verify facts and does "
        "not implement image analysis, social-network analysis, early-propagation "
        "detection, or a transformer."
    )
    values = metrics.get("test_metrics", {})
    cards = st.columns(4)
    cards[0].metric("Held-out test accuracy", f'{values.get("accuracy", 0):.3%}')
    cards[1].metric("Macro F1", f'{values.get("f1_macro", 0):.3%}')
    cards[2].metric("Test examples", f'{values.get("examples", 0):,}')
    cards[3].metric("Uncertain cutoff", f'{CONFIDENCE_THRESHOLD:.0%}')
    st.markdown(
        "**Model:** word TF-IDF + Logistic Regression  ·  "
        "**Dataset:** ISOT Fake News Dataset  ·  "
        "**Interface:** Streamlit"
    )


def render_analyzer(model) -> None:
    st.title("Analyze news text")
    st.caption(
        "Paste an English headline, article, or both. Headline-only inputs may have "
        "less evidence than the full articles used for training."
    )
    with st.form("analysis_form"):
        text = st.text_area(
            "News headline or article",
            height=220,
            max_chars=25_000,
            placeholder="Paste the news text here...",
        )
        submitted = st.form_submit_button("Analyze text", type="primary")
    if submitted:
        if not text.strip():
            st.info("Enter a headline or article first.")
        else:
            try:
                result = analyze_text(text, model=model)
                record_analysis(text, result)
                st.session_state["last_analysis"] = (text, result)
            except Exception as error:
                st.error(f"Could not analyze this text: {error}")
    current = st.session_state.get("last_analysis")
    if current:
        st.divider()
        show_result(current[1])


def render_news(model) -> None:
    st.title("Latest news headlines")
    st.caption(
        "Headlines come from public RSS feeds. They are unseen demo inputs, not model "
        "training data or independent verification."
    )
    if st.button("Refresh headlines", type="primary"):
        clear_news_cache()
    with st.spinner("Loading RSS feeds..."):
        articles, errors = fetch_latest_news()
    for error in errors:
        st.error(error)
    if not articles:
        st.info("No headlines are available. Check your internet connection and refresh.")
        return
    for index, article in enumerate(articles):
        with st.container(border=True):
            st.caption(f'{article["source"]} · {article["published"]}')
            st.markdown(f'**{article["title"]}**')
            if article["link"]:
                st.link_button("Open source article", article["link"])
            key = f'news-{index}-{article["title"][:30]}'
            if st.button("Analyze this headline", key=key):
                try:
                    result = analyze_text(article["title"], model=model)
                    st.session_state[f"result-{key}"] = result
                except Exception as error:
                    st.error(f"Could not analyze this headline: {error}")
            result = st.session_state.get(f"result-{key}")
            if result:
                show_result(result)


def render_performance(metrics: dict) -> None:
    st.title("Model performance")
    test = metrics.get("test_metrics")
    if not test:
        st.info("Training metrics are missing. Run python train_model.py first.")
        return
    st.caption(
        "All figures below are calculated on the held-out test split. The reported "
        "labels are inherited from the ISOT dataset files. Percentages are shown to "
        "three decimal places so closely sized metrics remain distinguishable."
    )
    cards = st.columns(4)
    cards[0].metric("Accuracy", f'{test["accuracy"]:.3%}')
    cards[1].metric("Macro precision", f'{test["precision_macro"]:.3%}')
    cards[2].metric("Macro recall", f'{test["recall_macro"]:.3%}')
    cards[3].metric("Macro F1", f'{test["f1_macro"]:.3%}')
    st.subheader("Confusion matrix")
    labels = test["confusion_matrix_labels"]
    matrix = pd.DataFrame(
        test["confusion_matrix"],
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )
    st.dataframe(matrix, use_container_width=True)
    st.caption(
        "Rows are actual dataset labels; columns are model predictions. This matrix "
        "uses the model's best class for every test item, before the UNCERTAIN cutoff."
    )
    a, b, c = st.columns(3)
    a.metric("Test coverage at cutoff", f'{test["test_coverage_at_threshold"]:.1%}')
    value = test.get("accuracy_on_confident_test_examples")
    b.metric("Accuracy when not uncertain", f"{value:.1%}" if value is not None else "N/A")
    c.metric("Uncertain test examples", f'{test["uncertain_test_examples"]:,}')
    st.subheader("Per-class results")
    class_rows = []
    for label, result in test["per_class"].items():
        class_rows.append({
            "Label": label,
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1"],
            "Test examples": result["support"],
        })
    st.dataframe(
        pd.DataFrame(class_rows).set_index("Label").style.format(
            {"Precision": "{:.3%}", "Recall": "{:.3%}", "F1": "{:.3%}"}
        ),
        use_container_width=True,
    )
    st.subheader("Evaluation details and limitations")
    st.write(f'**Dataset split:** {metrics.get("split", "Not recorded")}.')
    example_count = metrics.get("data_preparation", {}).get("prepared_examples", 0)
    st.write(f"**Examples after cleanup:** {example_count:,}.")
    st.caption(
        metrics.get(
            "evaluation_note",
            "These scores describe this held-out dataset only; they do not verify new claims.",
        )
    )


def render_history() -> None:
    st.title("Local analysis history")
    if not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0:
        st.info("No analyses have been saved on this computer yet.")
        return
    try:
        history = pd.read_csv(HISTORY_PATH)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        st.error("The history file could not be read.")
        return
    if history.empty:
        st.info("No analyses have been saved on this computer yet.")
    else:
        st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)
        st.caption("History is stored locally in history.csv. Submitted text is not uploaded.")


def render_about(metrics: dict) -> None:
    st.title("About this project")
    st.markdown(
        """
        **Prototype title:** Explainable Text-Based Fake News Classification Using
        TF-IDF and Logistic Regression.

        **Workflow:** cleaned title/article text → TF-IDF word features →
        Logistic Regression → class score → REAL / FAKE / UNCERTAIN.

        **Explainability:** for each input, the app multiplies its TF-IDF feature
        values by the trained Logistic Regression coefficients and displays the
        largest contributions in each direction.

        **Latest news:** public RSS headlines are fetched on demand and scored with
        the trained model. The feeds are not used for training or evaluation.
        """
    )
    st.markdown(
        f'**Training dataset:** {metrics.get("dataset", "ISOT Fake News Dataset")}. '
        "The original CSV files are used as provided; their labels are proxies based "
        "on source membership, not per-claim fact checks."
    )
    st.caption(
        "Limitations: English text only; no images or social context. Results may "
        "reflect publisher and writing-style patterns in ISOT, not claim truth."
    )


def main() -> None:
    st.sidebar.title("🔎 TruthLens")
    st.sidebar.caption("Explainable news classification")
    page = st.sidebar.radio("Pages", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("TF-IDF + Logistic Regression · English text")
    metrics = load_metrics()
    model = None
    model_error = None
    try:
        model = get_model()
    except Exception as error:
        model_error = str(error)
    ensure_history()

    if page == "Home":
        render_home(metrics, model is not None)
    elif page == "Analyze News":
        if model is None:
            st.error(f"Model is unavailable: {model_error}")
        else:
            render_analyzer(model)
    elif page == "Latest News":
        if model is None:
            st.error(f"Model is unavailable: {model_error}")
        else:
            render_news(model)
    elif page == "Model Performance":
        render_performance(metrics)
    elif page == "History":
        render_history()
    else:
        render_about(metrics)


if __name__ == "__main__":
    main()
