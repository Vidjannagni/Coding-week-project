"""Tests unitaires du pipeline de préparation des données."""

import os
import sys

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing import (
    load_data, optimize_memory, clean_data, preprocess_data,
    _impute_bmi, _impute_by_correlation,
)


                                                             
                       
                                                             

class TestLoadData:

    def test_loads_non_empty(self, raw_df):
        assert raw_df is not None
        assert len(raw_df) > 0

    def test_expected_row_count(self, raw_df):
        assert raw_df.shape[0] == 782

    def test_has_target_column(self, raw_df):
        assert "Diagnosis" in raw_df.columns

    def test_minimum_columns(self, raw_df):
        assert raw_df.shape[1] >= 50

    def test_has_key_features(self, raw_df):
        expected = ["Age", "BMI", "WBC_Count", "CRP", "Body_Temperature", "Sex"]
        for col in expected:
            assert col in raw_df.columns, f"Colonne attendue manquante : {col}"

    def test_diagnosis_values(self, raw_df):
        unique_vals = set(raw_df["Diagnosis"].dropna().unique())
        assert unique_vals.issubset({0, 1}), f"Valeurs inattendues dans Diagnosis : {unique_vals}"


                                                             
                             
                                                             

class TestOptimizeMemory:

    def test_reduces_memory(self, raw_df):
        mem_before = raw_df.memory_usage(deep=True).sum()
        df_opt = optimize_memory(raw_df)
        mem_after = df_opt.memory_usage(deep=True).sum()
        assert mem_after < mem_before

    def test_reduction_at_least_30_percent(self, raw_df):
        mem_before = raw_df.memory_usage(deep=True).sum()
        df_opt = optimize_memory(raw_df)
        mem_after = df_opt.memory_usage(deep=True).sum()
        reduction = (1 - mem_after / mem_before) * 100
        assert reduction > 30

    def test_preserves_row_count(self, raw_df, optimized_df):
        assert len(optimized_df) == len(raw_df)

    def test_preserves_columns(self, raw_df, optimized_df):
        assert list(optimized_df.columns) == list(raw_df.columns)

    def test_preserves_values(self, raw_df):
        df_opt = optimize_memory(raw_df)

        for col in raw_df.select_dtypes(include=[np.number]).columns[:5]:
            np.testing.assert_allclose(
                df_opt[col].values.astype(float),
                raw_df[col].values.astype(float),
                rtol=1e-3, atol=1e-5,
                err_msg=f"Valeurs différentes dans {col}"
            )

    def test_dtype_after_optimize(self, raw_df):
        df_opt = optimize_memory(raw_df)
        for col in df_opt.select_dtypes(include=[np.number]).columns:
            assert df_opt[col].dtype.itemsize <= raw_df[col].dtype.itemsize, (
                f"{col} : dtype optimisé ({df_opt[col].dtype}) "
                f"plus grand que l'original ({raw_df[col].dtype})"
            )


                                                             
                                  
                                                             

class TestImputeBMI:

    def test_calculates_bmi_from_height_weight(self):
        df = pd.DataFrame({
            "BMI": [np.nan, 25.0],
            "Weight": [70.0, 80.0],
            "Height": [175.0, 180.0],
        })
        result = _impute_bmi(df)
        expected_bmi = 70.0 / (1.75 ** 2)
        assert abs(result["BMI"].iloc[0] - expected_bmi) < 0.5
        assert result["BMI"].iloc[1] == 25.0                             

    def test_drops_height_weight(self):
        df = pd.DataFrame({
            "BMI": [25.0], "Weight": [70.0], "Height": [175.0],
        })
        result = _impute_bmi(df)
        assert "Weight" not in result.columns
        assert "Height" not in result.columns

    def test_no_bmi_column(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = _impute_bmi(df)
        assert list(result.columns) == ["A", "B"]

    def test_missing_weight_or_height(self):
        df = pd.DataFrame({
            "BMI": [np.nan], "Weight": [np.nan], "Height": [175.0],
        })
        result = _impute_bmi(df)
        assert pd.isna(result["BMI"].iloc[0])                        


                                                             
                                             
                                                             

class TestImputeByCorrelation:

    def test_imputes_correlated_nans(self):
        np.random.seed(42)
        n = 100
        a = np.random.randn(n) * 10 + 50
        b = a * 2 + np.random.randn(n) * 0.5                
        b[95:] = np.nan
        df = pd.DataFrame({"A": a, "B": b, "Diagnosis": np.random.randint(0, 2, n)})
        result = _impute_by_correlation(df, corr_threshold=0.5)
        assert result["B"].isna().sum() == 0

    def test_no_imputation_below_threshold(self):
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "A": np.random.randn(n),
            "B": np.random.randn(n),
            "Diagnosis": np.random.randint(0, 2, n),
        })
        df.loc[0, "B"] = np.nan
        result = _impute_by_correlation(df, corr_threshold=0.99)
        assert result["B"].isna().sum() == 1                                        


                                                             
                        
                                                             

class TestCleanData:

    def test_no_nans_in_numeric(self, clean_df):
        numeric = clean_df.select_dtypes(include=[np.number]).columns
        for col in numeric:
            assert clean_df[col].isna().sum() == 0, f"NaN subsiste dans {col}"

    def test_preserves_most_rows(self, raw_df, clean_df):
        assert len(clean_df) >= len(raw_df) * 0.85

    def test_removes_parasitic_columns(self, clean_df):
        removed = ["Management", "Severity", "Length_of_Stay",
                    "Alvarado_Score", "Paedriatic_Appendicitis_Score"]
        for col in removed:
            assert col not in clean_df.columns, f"{col} n'aurait pas dû survivre"

    def test_removes_weak_columns(self, clean_df):
        weak = ["Ketones_in_Urine", "RBC_in_Urine", "WBC_in_Urine",
                "Dysuria", "Stool", "US_Performed",
                "Hemoglobin", "RDW", "Thrombocyte_Count", "RBC_Count"]
        for col in weak:
            assert col not in clean_df.columns, f"{col} (faible) aurait dû être supprimé"

    def test_engineered_features_present(self, clean_df):
        assert "WBC_CRP_Ratio" in clean_df.columns

    def test_no_duplicates(self, clean_df):
        assert clean_df.duplicated().sum() == 0

    def test_diagnosis_still_present(self, clean_df):
        assert "Diagnosis" in clean_df.columns

    def test_height_weight_removed(self, clean_df):
        assert "Height" not in clean_df.columns
        assert "Weight" not in clean_df.columns

    def test_bmi_preserved(self, clean_df):
        assert "BMI" in clean_df.columns


                                                             
                             
                                                             

class TestPreprocessData:

    def test_shapes_match(self, preprocessed):
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_feature_names_count(self, preprocessed):
        X_train, _, _, _, _, feature_names = preprocessed
        assert len(feature_names) == X_train.shape[1]

    def test_target_binary(self, preprocessed):
        _, _, y_train, y_test, _, _ = preprocessed
        all_y = np.concatenate([y_train, y_test])
        assert set(all_y) == {0, 1}

    def test_split_ratio(self, preprocessed):
        X_train, X_test, _, _, _, _ = preprocessed
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert 0.15 <= test_ratio <= 0.25

    def test_stratification(self, preprocessed):
        _, _, y_train, y_test, _, _ = preprocessed
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05

    def test_scaler_fitted(self, preprocessed):
        _, _, _, _, scaler, _ = preprocessed
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "scale_")

    def test_train_approximately_scaled(self, preprocessed):
        X_train, _, _, _, _, _ = preprocessed
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0)
        assert np.allclose(means, 0, atol=0.15)
        assert np.allclose(stds, 1, atol=0.25)

    def test_no_nans_in_output(self, preprocessed):
        X_train, X_test, _, _, _, _ = preprocessed
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
