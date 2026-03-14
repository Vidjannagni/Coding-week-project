"""Point d'entrée du pipeline: préparation des données puis entraînement."""

import sys
import os

# Allow direct execution from src/ while keeping imports simple.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_processing import load_data, optimize_memory, clean_data, preprocess_data
from train_model import main as train


def run_pipeline():
    """Run end-to-end training and return all model results + best model name."""
    df = load_data()
    df = optimize_memory(df)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    # train_model expects already-prepared arrays and metadata.
    results, best_name = train(X_train, X_test, y_train, y_test, scaler, feature_names)

    return results, best_name


if __name__ == "__main__":
    run_pipeline()