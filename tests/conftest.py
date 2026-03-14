"""Fixtures partagées pour accélérer et stabiliser toute la suite de tests."""

import os
import sys

import pytest
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing import load_data, optimize_memory, clean_data, preprocess_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


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
