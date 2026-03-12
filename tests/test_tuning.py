"""Tests du tuning d'hyperparamètres et de l'évaluation des champions."""

import os
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tuning import (
    build_model, cv_score, grid_combinations,
    tune_all_models, evaluate_champions, final_test_score, GRIDS,
)


                                                             
                         
                                                             

class TestBuildModel:

    def test_svm(self):
        m = build_model("SVM", {"C": 1.0, "gamma": "scale", "class_weight": None})
        assert hasattr(m, "fit")
        assert hasattr(m, "predict_proba")

    def test_random_forest(self):
        m = build_model("Random Forest", {
            "n_estimators": 100, "max_depth": 10,
            "min_samples_split": 2, "min_samples_leaf": 1,
            "class_weight": None,
        })
        assert hasattr(m, "fit")

    def test_lightgbm(self):
        m = build_model("LightGBM", {
            "n_estimators": 50, "max_depth": 5,
            "learning_rate": 0.1, "num_leaves": 31,
            "class_weight": None,
        })
        assert hasattr(m, "fit")

    def test_catboost(self):
        m = build_model("CatBoost", {
            "iterations": 50, "depth": 6,
            "learning_rate": 0.1, "auto_class_weights": None,
        })
        assert hasattr(m, "fit")

    def test_all_models_have_predict_proba(self):
        configs = {
            "SVM": {"C": 1.0, "gamma": "scale", "class_weight": None},
            "Random Forest": {"n_estimators": 50, "max_depth": 5,
                              "min_samples_split": 2, "min_samples_leaf": 1,
                              "class_weight": None},
            "LightGBM": {"n_estimators": 50, "max_depth": 5,
                         "learning_rate": 0.1, "num_leaves": 31,
                         "class_weight": None},
            "CatBoost": {"iterations": 50, "depth": 5,
                         "learning_rate": 0.1, "auto_class_weights": None},
        }
        for name, params in configs.items():
            m = build_model(name, params)
            assert hasattr(m, "predict_proba"), f"{name} n'a pas predict_proba"


                                                             
                               
                                                             

class TestGridCombinations:

    def test_simple_grid(self):
        grid = {"a": [1, 2], "b": ["x", "y"]}
        combos = grid_combinations(grid)
        assert len(combos) == 4
        assert {"a": 1, "b": "x"} in combos
        assert {"a": 2, "b": "y"} in combos

    def test_single_param(self):
        combos = grid_combinations({"x": [10, 20, 30]})
        assert len(combos) == 3

    def test_empty_grid(self):
        combos = grid_combinations({})
        assert len(combos) == 1                                         

    def test_all_grids_have_combinations(self):
        for name, grid in GRIDS.items():
            combos = grid_combinations(grid)
            assert len(combos) > 0, f"Grille vide pour {name}"


                                                             
                      
                                                             

class TestCvScore:

    def test_returns_expected_keys(self, preprocessed):
        X_train, _, y_train, _, _, _ = preprocessed
        model = build_model("Random Forest", {
            "n_estimators": 50, "max_depth": 5,
            "min_samples_split": 2, "min_samples_leaf": 1,
            "class_weight": None,
        })
        scores = cv_score(model, X_train, y_train)
        assert "recall" in scores
        assert "precision" in scores
        assert "roc_auc" in scores
        assert "recall_std" in scores

    def test_scores_in_valid_range(self, preprocessed):
        X_train, _, y_train, _, _, _ = preprocessed
        model = build_model("Random Forest", {
            "n_estimators": 50, "max_depth": 5,
            "min_samples_split": 2, "min_samples_leaf": 1,
            "class_weight": None,
        })
        scores = cv_score(model, X_train, y_train)
        for key in ["recall", "precision", "roc_auc"]:
            assert 0 <= scores[key] <= 1, f"{key} hors intervalle"


                                                             
                              
                                                             

class TestFinalTestScore:

    def test_returns_metrics(self, preprocessed):
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        model = build_model("Random Forest", {
            "n_estimators": 50, "max_depth": 5,
            "min_samples_split": 2, "min_samples_leaf": 1,
            "class_weight": None,
        })
        scores = final_test_score(model, X_train, X_test, y_train, y_test)
        assert "recall" in scores
        assert "precision" in scores
        assert "roc_auc" in scores

    def test_scores_in_valid_range(self, preprocessed):
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        model = build_model("Random Forest", {
            "n_estimators": 50, "max_depth": 5,
            "min_samples_split": 2, "min_samples_leaf": 1,
            "class_weight": None,
        })
        scores = final_test_score(model, X_train, X_test, y_train, y_test)
        for key in ["recall", "precision", "roc_auc"]:
            assert 0 <= scores[key] <= 1


                                                             
                                                         
                                                             

class TestTuneAllModels:

    @pytest.fixture
    def mini_tune(self, preprocessed, monkeypatch):
                                                                         
        mini = {
            "SVM": {"C": [1.0], "gamma": ["scale"], "class_weight": [None]},
            "Random Forest": {"n_estimators": [50], "max_depth": [5],
                              "min_samples_split": [2], "min_samples_leaf": [1],
                              "class_weight": [None]},
        }
        monkeypatch.setattr("src.tuning.GRIDS", mini)
        X_train, _, y_train, _, _, _ = preprocessed
        return tune_all_models(X_train, y_train)

    def test_returns_dict(self, mini_tune):
        assert isinstance(mini_tune, dict)
        assert len(mini_tune) >= 2

    def test_each_model_has_best(self, mini_tune):
        for name, data in mini_tune.items():
            assert "best" in data
            assert "top3" in data
            assert "params" in data["best"]
            assert "cv_metrics" in data["best"]

    def test_cv_metrics_valid(self, mini_tune):
        for name, data in mini_tune.items():
            m = data["best"]["cv_metrics"]
            assert 0 <= m["recall"] <= 1
            assert 0 <= m["precision"] <= 1
            assert 0 <= m["roc_auc"] <= 1

    def test_top3_is_list(self, mini_tune):
        for name, data in mini_tune.items():
            assert isinstance(data["top3"], list), (
                f"top3 de {name} devrait être une liste"
            )


                                                             
                                
                                                             

class TestEvaluateChampions:

    def test_evaluate_champions(self, preprocessed, monkeypatch):
        mini = {
            "Random Forest": {"n_estimators": [50], "max_depth": [5],
                              "min_samples_split": [2], "min_samples_leaf": [1],
                              "class_weight": [None]},
        }
        monkeypatch.setattr("src.tuning.GRIDS", mini)
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        tuning_results = tune_all_models(X_train, y_train)
        final = evaluate_champions(tuning_results, X_train, X_test, y_train, y_test)
        assert isinstance(final, dict)
        for name, data in final.items():
            assert "best_params" in data
            assert "cv_metrics" in data
            assert "test_metrics" in data
            t = data["test_metrics"]
            assert 0 <= t["recall"] <= 1
            assert 0 <= t["precision"] <= 1
            assert 0 <= t["roc_auc"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
