"""Tests des métriques et graphiques d'évaluation du modèle."""

import os
import sys
import tempfile

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate_model import (
    load_model, evaluate_model, plot_confusion_matrix,
    plot_roc_curve, generate_shap_plots,
)


class TestLoadModel:

    def test_load_model_returns_tuple(self):
        model, scaler, feat = load_model()
        assert hasattr(model, "predict")
        assert hasattr(scaler, "transform")
        assert isinstance(feat, list)

    def test_feature_names_non_empty(self):
        _, _, feat = load_model()
        assert len(feat) > 0


class TestEvaluateModel:

    def test_evaluate_returns_metrics(self, preprocessed, loaded_model):
        _, X_test, _, y_test, _, _ = preprocessed
        model, _, _ = loaded_model
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert len(y_pred) == len(y_test)
        assert len(y_prob) == len(y_test)

    def test_metrics_in_valid_range(self, preprocessed, loaded_model):
        _, X_test, _, y_test, _, _ = preprocessed
        model, _, _ = loaded_model
        metrics, _, _ = evaluate_model(model, X_test, y_test)
        for k, v in metrics.items():
            assert 0 <= v <= 1, f"{k} hors intervalle : {v}"

    def test_predictions_binary(self, preprocessed, loaded_model):
        _, X_test, _, y_test, _, _ = preprocessed
        model, _, _ = loaded_model
        _, y_pred, _ = evaluate_model(model, X_test, y_test)
        assert set(y_pred).issubset({0, 1})

    def test_probabilities_sum_to_one(self, preprocessed, loaded_model):
        _, X_test, _, y_test, _, _ = preprocessed
        model, _, _ = loaded_model
        _, _, y_prob = evaluate_model(model, X_test, y_test)
        assert all(0 <= p <= 1 for p in y_prob)


class TestPlots:

    def test_confusion_matrix_saves(self, preprocessed, loaded_model):
        _, X_test, _, y_test, _, _ = preprocessed
        model, _, _ = loaded_model
        _, y_pred, _ = evaluate_model(model, X_test, y_test)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cm.png")
            plot_confusion_matrix(y_test, y_pred, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_roc_curve_saves(self, preprocessed, loaded_model):
        _, X_test, _, y_test, _, _ = preprocessed
        model, _, _ = loaded_model
        _, _, y_prob = evaluate_model(model, X_test, y_test)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "roc.png")
            plot_roc_curve(y_test, y_prob, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_shap_plots_save(self, preprocessed, loaded_model):
        _, X_test, _, y_test, _, _ = preprocessed
        model, _, feature_names = loaded_model
        with tempfile.TemporaryDirectory() as tmpdir:
            shap_values, explainer = generate_shap_plots(
                model, X_test, feature_names, save_dir=tmpdir
            )
            assert os.path.exists(os.path.join(tmpdir, "shap_summary_bar.png"))
            assert os.path.exists(os.path.join(tmpdir, "shap_beeswarm.png"))
            assert shap_values is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
