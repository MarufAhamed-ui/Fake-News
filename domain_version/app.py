"""HTML web application for the separately deployable TruthLens domain version."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from config import MAX_TEXT_CHARS
from news import clear_news_cache, fetch_latest_news
from predictor import analyze_text, load_metrics, load_model

BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 100_000
app.config["JSON_SORT_KEYS"] = False


@app.after_request
def add_response_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/healthz")
def health():
    try:
        load_model()
        return jsonify({"status": "ok", "model": "ready"})
    except Exception:
        app.logger.exception("Model health check failed")
        return jsonify({"status": "unavailable", "model": "not ready"}), 503


@app.get("/api/metrics")
def metrics():
    return jsonify(load_metrics())


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with a text field."}), 400
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Enter a headline or article first."}), 400
    if len(text) > MAX_TEXT_CHARS:
        return jsonify({"error": f"Text must be {MAX_TEXT_CHARS:,} characters or fewer."}), 413
    try:
        result = analyze_text(text)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "The trained model is missing. Add model.joblib to models/."}), 503
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "The text could not be analyzed."}), 500


@app.get("/api/news")
def news():
    refresh = request.args.get("refresh") == "1"
    if refresh:
        clear_news_cache()
    articles, errors = fetch_latest_news(force_refresh=refresh)
    return jsonify({"articles": articles, "errors": errors})


if __name__ == "__main__":
    # Local preview only. For a public host, use the WSGI command in README.md.
    app.run(host="127.0.0.1", port=5000, debug=False)
