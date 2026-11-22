"""CineScope backend.

Serves index.html and exposes two endpoints:
  GET  /api/options   dropdown values and ranges from model_options.json
  POST /api/predict   returns hit or flop, the hit probability and a confidence label

Run locally:       python app.py            then open http://127.0.0.1:5000
Run in production: gunicorn app:app

Keep these files in the same folder:
  app.py, index.html, best_movie_model.pkl, model_options.json
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).parent
app = Flask(__name__, static_folder=None)

with open(BASE / "best_movie_model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open(BASE / "model_options.json") as f:
    OPTIONS = json.load(f)

# request field -> key in model_options.json
CATEGORICAL = {
    "release_month": "release_month",
    "genre": "genre_grp",
    "rating": "rating_grp",
    "country": "country_grp",
    "company": "company_grp",
    "director": "director_grp",
    "star": "star_grp",
}

# request field -> (minimum, maximum), same limits as the old Streamlit sliders
NUMERIC = {
    "budget": (100_000, 500_000_000),
    "runtime": (OPTIONS["runtime_min"], 250),
    "year": (OPTIONS["year_min"], 2030),
    "score": (OPTIONS["score_min"], 10.0),
    "votes": (1_000, 3_000_000),
}


@app.after_request
def add_cors(resp):
    # Only needed when index.html is hosted on a different site than this API.
    # Set CORS_ORIGIN to that site's address in production, for example
    # https://yourname.github.io
    resp.headers["Access-Control-Allow-Origin"] = os.environ.get("CORS_ORIGIN", "*")
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/api/options")
def options():
    return jsonify(OPTIONS)


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    clean = {}

    for field, (low, high) in NUMERIC.items():
        try:
            value = float(data[field])
        except (KeyError, TypeError, ValueError):
            return jsonify(error=f"Enter a number for {field}."), 400
        if not low <= value <= high:
            return jsonify(error=f"{field} must be between {low:,} and {high:,}."), 400
        clean[field] = value

    for field, key in CATEGORICAL.items():
        value = data.get(field)
        if value not in OPTIONS[key]:
            return jsonify(error=f"Choose a valid value for {field}."), 400
        clean[field] = value

    # Same feature set and order the model was trained on (see the old app.py)
    row = pd.DataFrame([{
        "budget": clean["budget"],
        "log_budget": np.log1p(clean["budget"]),
        "runtime": clean["runtime"],
        "budget_per_min": clean["budget"] / clean["runtime"],
        "year": clean["year"],
        "score": clean["score"],
        "votes": clean["votes"],
        "log_votes": np.log1p(clean["votes"]),
        "release_month": clean["release_month"],
        "genre_grp": clean["genre"],
        "rating_grp": clean["rating"],
        "country_grp": clean["country"],
        "company_grp": clean["company"],
        "director_grp": clean["director"],
        "star_grp": clean["star"],
    }])

    prob = float(MODEL.predict_proba(row)[0][1])
    pred = int(MODEL.predict(row)[0])

    gap = abs(prob - 0.5)
    if gap >= 0.30:
        confidence = "High confidence"
    elif gap >= 0.15:
        confidence = "Moderate confidence"
    else:
        confidence = "Close call"

    return jsonify(
        prediction="hit" if pred == 1 else "flop",
        probability=prob,
        confidence=confidence,
    )


if __name__ == "__main__":
    app.run(debug=True)
