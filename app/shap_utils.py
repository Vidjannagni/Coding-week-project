"""Utilitaires SHAP pour initialiser l'explainer et formater les contributions."""

import logging

import shap
from config import (
    SHAP_TOP_N, SHAP_BG_SAMPLES,
    TREE_MODEL_TYPES, VOTING_PREFERRED_ESTIMATORS,
)

logger = logging.getLogger(__name__)

# Labels used in the UI to present features in French.
FEATURE_NAMES_FR = {
    "Age": "Âge",
    "BMI": "IMC",
    "Height": "Taille",
    "Weight": "Poids",
    "Sex_male": "Sexe (masculin)",
    "Body_Temperature": "Température corporelle",
    "WBC_Count": "Globules blancs (GB)",
    "Neutrophil_Percentage": "Pourcentage neutrophiles",
    "CRP": "Protéine C-réactive",
    "RBC_Count": "Globules rouges (GR)",
    "Hemoglobin": "Hémoglobine",
    "RDW": "IDR (Indice distrib. GR)",
    "Thrombocyte_Count": "Plaquettes",
    "Appendix_Diameter": "Diamètre appendice",
    "Length_of_Stay": "Durée de séjour",
    "Alvarado_Score": "Score d'Alvarado",
    "Paedriatic_Appendicitis_Score": "Score péd. appendicite",
    "WBC_CRP_Ratio": "Ratio GB / CRP",
    "Neutrophil_WBC_Interaction": "Interaction neutro. × GB",
    "Migratory_Pain_yes": "Douleur migratoire",
    "Lower_Right_Abd_Pain_yes": "Douleur fosse iliaque droite",
    "Contralateral_Rebound_Tenderness_yes": "Rebond controlatéral",
    "Coughing_Pain_yes": "Douleur à la toux",
    "Nausea_yes": "Nausées",
    "Loss_of_Appetite_yes": "Perte d'appétit",
    "Neutrophilia_yes": "Neutrophilie",
    "Dysuria_yes": "Dysurie",
    "Psoas_Sign_yes": "Signe du psoas",
    "Ipsilateral_Rebound_Tenderness_yes": "Rebond ipsilatéral",
    "US_Performed_yes": "Échographie réalisée",
    "Appendix_on_US_yes": "Appendice visible (écho)",
    "Free_Fluids_yes": "Liquide libre (écho)",
    "Peritonitis_no": "Péritonite absente",
    "Peritonitis_local": "Péritonite locale",
    "Stool_normal": "Selles normales",
    "Stool_diarrhea": "Diarrhée",
    "Stool_constipation, diarrhea": "Constipation",
    "Ketones_in_Urine_no": "Cétones urinaires (non)",
    "Ketones_in_Urine_++": "Cétones urinaires (++)",
    "Ketones_in_Urine_+++": "Cétones urinaires (+++)",
    "RBC_in_Urine_no": "GR urinaires (non)",
    "RBC_in_Urine_++": "GR urinaires (++)",
    "RBC_in_Urine_+++": "GR urinaires (+++)",
    "WBC_in_Urine_no": "GB urinaires (non)",
    "WBC_in_Urine_++": "GB urinaires (++)",
    "WBC_in_Urine_+++": "GB urinaires (+++)",
}


def init_explainer(model):
    """Pick the most compatible SHAP explainer for the current classifier."""
    model_type = type(model).__name__
    logger.info("Init SHAP explainer | model_type=%s", model_type)

    if model_type == "VotingClassifier":
        for name in VOTING_PREFERRED_ESTIMATORS:
            sub = model.named_estimators_.get(name)
            if sub is not None:
                logger.info("SHAP explainer using sub-estimator '%s' (%s)",
                            name, type(sub).__name__)
                return shap.TreeExplainer(sub)
        return shap.TreeExplainer(model.estimators_[0])

    if model_type in TREE_MODEL_TYPES:
        logger.info("SHAP TreeExplainer selected for model type %s", model_type)
        return shap.TreeExplainer(model)

    logger.info("No prebuilt explainer for %s, fallback to KernelExplainer at predict time", model_type)
    return None


def compute_shap_values(X_scaled, model, feature_names, explainer):
    """Compute SHAP values for one sample and return the top features for UI."""
    if explainer is None:
        logger.info("Using KernelExplainer fallback | bg_samples=%d", min(SHAP_BG_SAMPLES, len(X_scaled)))
        bg = shap.sample(X_scaled, min(SHAP_BG_SAMPLES, len(X_scaled)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)

    sv = explainer.shap_values(X_scaled)
    if isinstance(sv, list):
        # Binary classifier: keep positive class contribution.
        sv = sv[1]
    elif hasattr(sv, 'ndim') and sv.ndim == 3:
        # Shape (n_samples, n_features, n_classes) -> class 1.
        sv = sv[:, :, 1]

    vals = sv[0]
    pairs = list(zip(feature_names, X_scaled[0], vals))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    logger.debug("SHAP computed | features=%d | top_feature=%s", len(feature_names), pairs[0][0] if pairs else "n/a")

    return [{"feature": FEATURE_NAMES_FR.get(f, f),
             "value": round(float(v), 4),
             "shap": round(float(s), 4)}
            for f, v, s in pairs[:SHAP_TOP_N]]
