"""
app.py
======
Flask API serving the trained house-price model.

Endpoints
---------
GET  /          — service banner
GET  /health    — liveness probe (returns model status)
POST /predict   — single-record prediction; JSON in, JSON out
POST /predict/batch — batch prediction; takes a list of records

Run
---
    python app.py
    # → listens on http://0.0.0.0:5000

Example request
---------------
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{
               "POSTED_BY": "Owner",
               "UNDER_CONSTRUCTION": 0,
               "RERA": 0,
               "BHK_NO.": 2,
               "BHK_OR_RK": "BHK",
               "SQUARE_FT": 1300,
               "READY_TO_MOVE": 1,
               "RESALE": 1,
               "ADDRESS": "Ksfc Layout,Bangalore",
               "LONGITUDE": 12.96991,
               "LATITUDE": 77.59796
             }'

Example response
----------------
    {
      "predicted_price_lacs": 56.42,
      "predicted_price_inr":  5642000.0,
      "currency": "INR",
      "model_version": "rf-v1"
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# Import the feature-engineering helpers from the same module the trainer
# uses, so the API applies *exactly* the same transforms the model was
# trained on. This avoids the classic "training/serving skew" footgun, and
# also gives joblib the correct path to TargetEncoder for un-pickling.
from preprocessing import TargetEncoder, engineer_features  # noqa: F401
from train import FEATURE_COLUMNS

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
META_PATH = HERE / "feature_columns.json"

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run `python train.py` first."
    )

MODEL = joblib.load(MODEL_PATH)
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

# Columns the API expects in the *raw* input. Engineered features
# (log_sqft / sqft_per_bhk / city / is_metro) are added server-side.
RAW_INPUT_COLUMNS = [
    "POSTED_BY", "UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "BHK_OR_RK",
    "SQUARE_FT", "READY_TO_MOVE", "RESALE", "ADDRESS", "LONGITUDE", "LATITUDE",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_record(rec: dict) -> tuple[bool, str | None]:
    """Lightweight schema check on a single input record."""
    if not isinstance(rec, dict):
        return False, "Each record must be a JSON object."
    missing = [c for c in RAW_INPUT_COLUMNS if c not in rec]
    if missing:
        return False, f"Missing fields: {missing}"
    return True, None


def _predict_df(df: pd.DataFrame) -> np.ndarray:
    """Apply feature engineering then call the saved pipeline.

    The pipeline was trained on log(price), so we expm1 back to lacs.
    """
    df = engineer_features(df)
    log_pred = MODEL.predict(df[FEATURE_COLUMNS])
    return np.expm1(log_pred)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "house-price-prediction-api",
        "endpoints": ["/health", "/predict", "/predict/batch"],
        "model_metrics": META.get("metrics", {}).get("final"),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """Predict the price for a single property.

    Body: a JSON object with all RAW_INPUT_COLUMNS keys.
    Returns: predicted price in lacs and INR.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    ok, err = _validate_record(payload)
    if not ok:
        return jsonify({"error": err}), 400

    try:
        df = pd.DataFrame([payload], columns=RAW_INPUT_COLUMNS)
        price_lacs = float(_predict_df(df)[0])
    except Exception as e:  # pragma: no cover — defensive
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({
        "predicted_price_lacs": round(price_lacs, 2),
        "predicted_price_inr":  round(price_lacs * 100_000, 2),
        "currency": "INR",
        "model_version": "rf-v1",
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Predict prices for many properties in one call.

    Body: {"records": [<obj>, <obj>, ...]}
    Returns: {"predictions": [{"predicted_price_lacs": ...}, ...]}
    """
    payload = request.get_json(silent=True) or {}
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return jsonify({"error": "Body must include non-empty 'records' list."}), 400

    for i, rec in enumerate(records):
        ok, err = _validate_record(rec)
        if not ok:
            return jsonify({"error": f"Record {i}: {err}"}), 400

    try:
        df = pd.DataFrame(records, columns=RAW_INPUT_COLUMNS)
        prices = _predict_df(df)
    except Exception as e:  # pragma: no cover
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({
        "predictions": [
            {
                "predicted_price_lacs": round(float(p), 2),
                "predicted_price_inr":  round(float(p) * 100_000, 2),
            }
            for p in prices
        ],
        "count": len(prices),
        "currency": "INR",
        "model_version": "rf-v1",
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # debug=False so it's drop-in for production / Docker.
    # Set FLASK_DEBUG=1 in the environment for development.
    import os
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug)
