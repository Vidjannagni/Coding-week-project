"""Configuration centrale du pipeline ML (données, nettoyage, plots, SHAP)."""

import os

# Project paths.
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
FIGURES_DIR  = os.path.join(BASE_DIR, "reports", "figures")
IMAGES_DIR   = os.path.join(BASE_DIR, "reports", "images")
RESULTS_FILE = os.path.join(MODELS_DIR, "param_search_results.json")

# Shared ML defaults.
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# Data-cleaning thresholds.
MISSING_THRESHOLD        = 0.50
CORR_THRESHOLD           = 0.69
IQR_FACTOR               = 1.5
MIN_CORR_TRAIN_SIZE      = 5
HEIGHT_CM_THRESHOLD      = 3.0
CATEGORY_RATIO_THRESHOLD = 0.5

# Target column used across training and evaluation.
TARGET_COL = "Diagnosis"

# Groups dropped during cleaning.
TARGET_COLS  = ["Management", "Severity"]
LEAKAGE_COLS = ["Length_of_Stay"]
CIRCULAR_COLS = ["Alvarado_Score", "Paedriatic_Appendicitis_Score"]
WEAK_COLS = [
    "Ketones_in_Urine", "RBC_in_Urine", "WBC_in_Urine",
    "Dysuria", "Stool", "US_Performed",
    "Hemoglobin", "RDW", "Thrombocyte_Count", "RBC_Count",
]

# Plot defaults.
PLOT_DPI          = 150
PLOT_FIGSIZE_STD  = (8, 6)
PLOT_FIGSIZE_WIDE = (10, 8)

# SHAP defaults.
SHAP_MAX_DISPLAY = 15
SHAP_BG_SAMPLES  = 50

# Model families with native TreeExplainer support.
TREE_MODEL_TYPES            = ("RandomForestClassifier", "LGBMClassifier", "CatBoostClassifier")
VOTING_PREFERRED_ESTIMATORS = ("lgbm", "cat", "rf")
