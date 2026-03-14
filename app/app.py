"""Application Flask principale: formulaires, prédiction et rendu des résultats."""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from config import (
    MODELS_DIR, DEFAULT_SECRET_KEY,
    NUMERIC_FEATURES, BINARY_FEATURE_MAP, WBC_CRP_SMOOTHING,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", DEFAULT_SECRET_KEY)

from auth import auth_bp, setup_login_manager, init_db, get_db
from shap_utils import init_explainer, compute_shap_values

setup_login_manager(app)
app.register_blueprint(auth_bp)
init_db()

model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
explainer = init_explainer(model)
logger.info("Application init OK | model=%s | features=%d", type(model).__name__, len(feature_names))


                                

def build_feature_vector(form_data):
    """Convert form data into the exact feature dictionary expected by the model."""
    logger.debug("Building feature vector from %d form fields", len(form_data))
    vec = {}
    for col in NUMERIC_FEATURES:
        val = form_data.get(col, "0")
        vec[col] = float(val) if val else 0.0
    vec["WBC_CRP_Ratio"] = vec["WBC_Count"] / (vec["CRP"] + WBC_CRP_SMOOTHING)
    for form_key, (feat, positive) in BINARY_FEATURE_MAP.items():
        vec[feat] = 1 if form_data.get(form_key, "").lower() == positive else 0
    # Keep the same one-hot convention as training.
    perit = form_data.get("Peritonitis", "no").lower()
    vec["Peritonitis_local"] = 1 if perit == "local" else 0
    vec["Peritonitis_no"] = 1 if perit == "no" else 0
    logger.debug("Feature vector built with %d keys", len(vec))
    return vec


def prepare_input(feature_vector):
    """Align one sample with training columns and apply the saved scaler."""
    df = pd.DataFrame([feature_vector])
    missing = 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
            missing += 1
    df = df[feature_names]
    logger.debug("Input prepared | shape=%s | missing_filled=%d", df.shape, missing)
    return scaler.transform(df.values)


                

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/diagnosis")
@login_required
def diagnosis():
    return render_template("diagnosis.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """Run model inference, compute SHAP explanation, and persist history."""
    try:
        form_data = request.form.to_dict()
        logger.info("Predict request | user_id=%s | form_fields=%d", current_user.id, len(form_data))
        vec = build_feature_vector(form_data)
        X_scaled = prepare_input(vec)

        prediction = int(model.predict(X_scaled)[0])
        proba = model.predict_proba(X_scaled)[0]
        probability = float(proba[1])
        confidence = float(max(proba))
        logger.info(
            "Prediction complete | user_id=%s | pred=%d | proba=%.4f | confidence=%.4f",
            current_user.id,
            prediction,
            probability,
            confidence,
        )

        shap_values = compute_shap_values(X_scaled, model, feature_names, explainer)

        # Store the prediction payload for user history and admin review.
        conn = get_db()
        cursor = conn.execute(
            "INSERT INTO history (user_id, timestamp, age, sex, prediction, confidence, probability, form_data, patient_first_name, patient_last_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (current_user.id,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             form_data.get("Age", ""),
             form_data.get("Sex", ""),
             prediction,
             confidence,
             probability,
             json.dumps(form_data),
             form_data.get("patient_first_name", "").strip(),
             form_data.get("patient_last_name", "").strip())
        )
        conn.commit()
        conn.close()
        logger.info("Prediction persisted | user_id=%s | history_id=%s", current_user.id, cursor.lastrowid)

        return render_template("result.html",
                               prediction=prediction,
                               probability=probability,
                               confidence=confidence,
                               shap_values=shap_values,
                               form_data=form_data,
                               patient_name=f"{form_data.get('patient_first_name','').strip()} {form_data.get('patient_last_name','').strip()}".strip())
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        flash(f"Error: {e}", "error")
        return redirect(url_for("diagnosis"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
