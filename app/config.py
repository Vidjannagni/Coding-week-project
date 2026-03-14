"""Constantes de l'application Flask (UI, auth, SHAP, chemins)."""

import os

# Paths used by the Flask app and model loading.
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH    = os.path.join(APP_DIR, "history.db")

# Default secret key for local/dev usage.
DEFAULT_SECRET_KEY = "pediappend-secret-key-change-in-prod"

# Auth policy.
MIN_USERNAME_LENGTH    = 3
MIN_PASSWORD_LENGTH    = 6
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"

# Numeric fields expected in the diagnosis form.
NUMERIC_FEATURES = [
    "Age", "BMI", "Appendix_Diameter",
    "Body_Temperature", "WBC_Count", "CRP",
]

# Epsilon to avoid division by zero in WBC/CRP ratio.
WBC_CRP_SMOOTHING = 0.1

# Mapping from form input -> model one-hot feature.
BINARY_FEATURE_MAP = {
    "Sex":                              ("Sex_male", "male"),
    "Migratory_Pain":                   ("Migratory_Pain_yes", "yes"),
    "Lower_Right_Abd_Pain":             ("Lower_Right_Abd_Pain_yes", "yes"),
    "Contralateral_Rebound_Tenderness": ("Contralateral_Rebound_Tenderness_yes", "yes"),
    "Coughing_Pain":                    ("Coughing_Pain_yes", "yes"),
    "Nausea":                           ("Nausea_yes", "yes"),
    "Loss_of_Appetite":                 ("Loss_of_Appetite_yes", "yes"),
    "Neutrophilia":                     ("Neutrophilia_yes", "yes"),
    "Psoas_Sign":                       ("Psoas_Sign_yes", "yes"),
    "Ipsilateral_Rebound_Tenderness":   ("Ipsilateral_Rebound_Tenderness_yes", "yes"),
    "Appendix_on_US":                   ("Appendix_on_US_yes", "yes"),
    "Free_Fluids":                      ("Free_Fluids_yes", "yes"),
}

# SHAP display defaults.
SHAP_TOP_N                  = 10                                                  
SHAP_BG_SAMPLES             = 50                                                         
TREE_MODEL_TYPES            = ("RandomForestClassifier", "LGBMClassifier", "CatBoostClassifier")
VOTING_PREFERRED_ESTIMATORS = ("lgbm", "cat", "rf")
