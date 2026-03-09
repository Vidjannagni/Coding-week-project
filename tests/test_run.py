"""Tests d'intégration légers du pipeline complet de préparation des données."""

import os
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing import load_data, optimize_memory, clean_data, preprocess_data


class TestPipelineIntegrity:
    """Checks de cohérence entre plusieurs exécutions du pipeline."""

    def test_full_pipeline(self, raw_df):
        """Le pipeline doit produire des tableaux sans NaN."""
        df_opt = optimize_memory(raw_df)
        df_clean = clean_data(df_opt)
        X_train, X_test, y_train, y_test, scaler, feat = preprocess_data(df_clean)
        assert X_train.shape[0] + X_test.shape[0] == len(df_clean)
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_reproducibility(self, raw_df):
        """Deux exécutions avec le même random_state doivent matcher."""
        df1 = clean_data(optimize_memory(raw_df))
        df2 = clean_data(optimize_memory(raw_df))
        X1, _, y1, _, _, f1 = preprocess_data(df1)
        X2, _, y2, _, _, f2 = preprocess_data(df2)
        assert f1 == f2
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_feature_names_match_model(self, preprocessed, loaded_model):
                                                                                   
        _, _, _, _, _, pipeline_features = preprocessed
        _, _, model_features = loaded_model
        assert pipeline_features == model_features

    def test_pipeline_output_types(self, preprocessed):
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocessed
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(feature_names, list)
        assert hasattr(scaler, "transform")

    def test_feature_count_is_21(self, preprocessed):
        """Le pipeline doit produire exactement 21 features après nettoyage."""
        _, _, _, _, _, feature_names = preprocessed
        assert len(feature_names) == 21, (
            f"Nombre de features attendu : 21, obtenu : {len(feature_names)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
