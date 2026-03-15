"""Fixtures partagées pour accélérer et stabiliser toute la suite de tests."""

import os
import sys
import json
import warnings

import pytest
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing import load_data, optimize_memory, clean_data, preprocess_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def pytest_configure(config):
    """Suppress known third-party warnings that are not actionable."""
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*NumPy global RNG.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*does not have valid feature names.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*has feature names, but .* was fitted without feature names.*",
    )


@pytest.fixture(scope="session")
def raw_df():
    """Load the raw dataset once per session."""
    return load_data()


@pytest.fixture(scope="session")
def optimized_df(raw_df):
    return optimize_memory(raw_df)


@pytest.fixture(scope="session")
def clean_df(optimized_df):
    return clean_data(optimized_df)


@pytest.fixture(scope="session")
def preprocessed(clean_df):
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(clean_df)
    return X_train, X_test, y_train, y_test, scaler, feature_names


@pytest.fixture(scope="session")
def loaded_model():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return model, scaler, feature_names


@pytest.fixture(scope="session")
def metrics_data():
    """Load metrics.json once per session."""
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def y_pred_test(preprocessed, loaded_model):
    """Prédictions binaires du meilleur modèle sur le jeu de test."""
    _, X_test, _, _, _, _ = preprocessed
    model, _, _ = loaded_model
    return model.predict(X_test)


@pytest.fixture(scope="session")
def y_prob_test(preprocessed, loaded_model):
    """Probabilités de classe positive du meilleur modèle sur le jeu de test."""
    _, X_test, _, _, _, _ = preprocessed
    model, _, _ = loaded_model
    return model.predict_proba(X_test)[:, 1]
