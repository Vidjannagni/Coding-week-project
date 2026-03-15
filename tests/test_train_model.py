"""Tests du module d'entraînement: sélection, métriques et artefacts."""

import os
import sys
import json
import tempfile

import warnings

import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train_model import get_models, train_and_evaluate, select_best_model, save_artifacts

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


                                                             
                                                               
                                                             

class TestTrainModel:

    def test_get_models_returns_dict(self):
        models = get_models()
        assert isinstance(models, dict)
        assert len(models) >= 3

    def test_get_models_known_names(self):
        models = get_models()
        assert "Random Forest" in models
        assert "SVM" in models

    def test_all_models_have_fit_predict(self):
        models = get_models()
        for name, m in models.items():
            assert hasattr(m, "fit"), f"{name} n'a pas de méthode fit"
            assert hasattr(m, "predict"), f"{name} n'a pas de méthode predict"
            assert hasattr(m, "predict_proba"), f"{name} n'a pas de predict_proba"

    def test_train_and_evaluate(self, preprocessed):
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        results = train_and_evaluate(X_train, X_test, y_train, y_test)
        assert isinstance(results, dict)
        assert len(results) >= 3
        for name, data in results.items():
            m = data["metrics"]
            assert 0 <= m["accuracy"] <= 1
            assert 0 <= m["precision"] <= 1
            assert 0 <= m["recall"] <= 1
            assert 0 <= m["f1_score"] <= 1
            assert 0 <= m["roc_auc"] <= 1
            assert "model" in data

    def test_select_best_model(self, preprocessed):
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        results = train_and_evaluate(X_train, X_test, y_train, y_test)
        best_name, best_model, best_metrics = select_best_model(results)
        assert best_name in results
        assert hasattr(best_model, "predict")
        assert best_metrics["recall"] > 0.5                         

    def test_save_artifacts(self, preprocessed):
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocessed
        results = train_and_evaluate(X_train, X_test, y_train, y_test)
        best_name, best_model, _ = select_best_model(results)

        with tempfile.TemporaryDirectory() as tmpdir:
            import src.train_model as tm
            original_dir = tm.MODELS_DIR
            tm.MODELS_DIR = tmpdir
            try:
                save_artifacts(best_model, scaler, feature_names, best_name, results)
                assert os.path.exists(os.path.join(tmpdir, "best_model.pkl"))
                assert os.path.exists(os.path.join(tmpdir, "scaler.pkl"))
                assert os.path.exists(os.path.join(tmpdir, "feature_names.pkl"))
                assert os.path.exists(os.path.join(tmpdir, "metrics.json"))

                with open(os.path.join(tmpdir, "metrics.json")) as f:
                    saved = json.load(f)
                assert saved["best_model"] == best_name
                assert "models" in saved
            finally:
                tm.MODELS_DIR = original_dir


                                                             
                                                          
                                                             

class TestSavedModel:

    def test_model_files_exist(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))
        assert os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl"))
        assert os.path.exists(os.path.join(MODELS_DIR, "feature_names.pkl"))
        assert os.path.exists(os.path.join(MODELS_DIR, "metrics.json"))

    def test_feature_names_count(self, loaded_model):
        _, _, feature_names = loaded_model
        assert len(feature_names) == 21

    def test_expected_features(self, loaded_model):
        _, _, feature_names = loaded_model
        expected = [
            "Age", "BMI", "Appendix_Diameter", "Body_Temperature",
            "WBC_Count", "CRP", "WBC_CRP_Ratio", "Sex_male",
        ]
        for f in expected:
            assert f in feature_names, f"Feature attendue manquante : {f}"

    def test_prediction_valid(self, loaded_model):
        model, scaler, feature_names = loaded_model
        dummy = np.zeros((1, len(feature_names)))
        X = pd.DataFrame(scaler.transform(dummy), columns=feature_names)
        pred = model.predict(X)
        proba = model.predict_proba(X)
        assert pred[0] in [0, 1]
        assert proba.shape == (1, 2)
        assert abs(proba[0].sum() - 1.0) < 1e-5

    def test_probabilities_in_range(self, loaded_model):
        model, scaler, feature_names = loaded_model
        dummy = np.zeros((1, len(feature_names)))
        X = pd.DataFrame(scaler.transform(dummy), columns=feature_names)
        proba = model.predict_proba(X)
        assert 0 <= proba[0][0] <= 1
        assert 0 <= proba[0][1] <= 1

    def test_metrics_json_valid(self):
        with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
            data = json.load(f)
        assert "best_model" in data
        assert "models" in data
        assert isinstance(data["models"], dict)
        for name, metrics in data["models"].items():
            assert "accuracy" in metrics
            assert "f1_score" in metrics
            assert "roc_auc" in metrics

    def test_model_performance_acceptable(self):
        with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
            data = json.load(f)
        best = data["best_model"]
        m = data["models"][best]
        assert m["accuracy"] >= 0.85, f"Accuracy trop basse : {m['accuracy']}"
        assert m["recall"] >= 0.85, f"Recall trop bas : {m['recall']}"
        assert m["roc_auc"] >= 0.85, f"ROC-AUC trop bas : {m['roc_auc']}"

    def test_precision_above_threshold(self):
        with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
            data = json.load(f)
        best = data["best_model"]
        m = data["models"][best]
        assert m["precision"] >= 0.80, f"Precision trop basse : {m['precision']}"

    def test_f1_score_acceptable(self):
        with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
            data = json.load(f)
        best = data["best_model"]
        m = data["models"][best]
        assert m["f1_score"] >= 0.85, f"F1-score trop bas : {m['f1_score']}"

    def test_metrics_json_has_all_models(self):
        with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
            data = json.load(f)
        known_models = get_models()
        for name in known_models:
            assert name in data["models"], f"Modèle absent de metrics.json : {name}"

    def test_batch_prediction(self, loaded_model):
        model, scaler, feature_names = loaded_model
        batch = np.random.randn(10, len(feature_names))
        X = pd.DataFrame(scaler.transform(batch), columns=feature_names)
        preds = model.predict(X)
        probas = model.predict_proba(X)
        assert preds.shape == (10,)
        assert probas.shape == (10, 2)
        assert all(p in [0, 1] for p in preds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
