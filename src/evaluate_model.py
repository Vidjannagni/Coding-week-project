
"""Évaluation du modèle: métriques, courbes, matrice de confusion et graphiques SHAP."""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_processing import load_data, optimize_memory, clean_data, preprocess_data
from src.config import (
    MODELS_DIR, IMAGES_DIR,
    PLOT_DPI, PLOT_FIGSIZE_STD, PLOT_FIGSIZE_WIDE,
    SHAP_MAX_DISPLAY, SHAP_BG_SAMPLES, TREE_MODEL_TYPES,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model():
    """Load the production model, scaler, and ordered feature list."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    logger.info("Model loaded | type=%s | features=%d", type(model).__name__, len(feature_names))
    return model, scaler, feature_names


def evaluate_model(model, X_test, y_test):
    """Compute classification metrics and return metrics + predictions."""
    logger.info("Evaluation start | X_test=%s | y_test=%d", X_test.shape, len(y_test))
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Appendicitis", "Appendicitis"]))
    logger.info("Evaluation done | metrics=%s", metrics)

    return metrics, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """Generate and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=PLOT_FIGSIZE_STD)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Appendicitis", "Appendicitis"],
        yticklabels=["No Appendicitis", "Appendicitis"],
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)
    plt.close()


def plot_roc_curve(y_test, y_prob, save_path=None):
    """Generate and optionally save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=PLOT_FIGSIZE_STD)
    plt.plot(fpr, tpr, color="#6366f1", lw=2.5, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random")
    plt.fill_between(fpr, tpr, alpha=0.15, color="#6366f1")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info("ROC curve saved to %s", save_path)
    plt.close()


def generate_shap_plots(model, X_test, feature_names: list, save_dir: str = IMAGES_DIR):
    """Generate SHAP summary and waterfall plots and save them to disk."""
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Generating SHAP explanations | model=%s | samples=%d", type(model).__name__, len(X_test))

    # Tree models can use TreeExplainer directly.
    model_type = type(model).__name__
    if model_type in TREE_MODEL_TYPES:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # For binary tasks SHAP may return one array per class.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
                                # Non-tree models fallback to KernelExplainer with a background sample.
        bg = shap.sample(X_test, min(SHAP_BG_SAMPLES, len(X_test)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X_test[:100])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    # Use DataFrame to keep readable feature labels in SHAP charts.
    X_df = pd.DataFrame(X_test, columns=feature_names)

    # Summary bar plot.
    plt.figure(figsize=PLOT_FIGSIZE_WIDE)
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False, max_display=SHAP_MAX_DISPLAY)
    plt.title("SHAP Feature Importance (Top 15)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary_bar.png"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary bar plot saved.")

    # Beeswarm plot.
    plt.figure(figsize=PLOT_FIGSIZE_WIDE)
    shap.summary_plot(shap_values, X_df, show=False, max_display=SHAP_MAX_DISPLAY)
    plt.title("SHAP Beeswarm Plot (Top 15)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_beeswarm.png"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info("SHAP beeswarm plot saved.")

    # Waterfall plot for the first sample.
    try:
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1],
            data=X_test[0],
            feature_names=feature_names,
        )
        plt.figure(figsize=PLOT_FIGSIZE_WIDE)
        shap.waterfall_plot(explanation, show=False, max_display=SHAP_MAX_DISPLAY)
        plt.title("SHAP Waterfall (Sample Prediction)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_waterfall.png"), dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        logger.info("SHAP waterfall plot saved.")
    except Exception as e:
        logger.warning("Could not generate waterfall plot: %s", e)

    return shap_values, explainer


def generate_single_prediction_shap(model, scaler, feature_names, input_data: dict, save_path: str):
    """Generate one SHAP waterfall figure for a single patient prediction."""
    import io
    import base64

    # Align incoming payload to training feature order.
    df_input = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_names]
    X_scaled = scaler.transform(df_input.values)

    # Rebuild a local explainer for one-sample explanation.
    model_type = type(model).__name__
    if model_type in TREE_MODEL_TYPES:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    else:
        bg = shap.sample(X_scaled, 1)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=X_scaled[0],
        feature_names=feature_names,
    )

    plt.figure(figsize=PLOT_FIGSIZE_WIDE)
    shap.waterfall_plot(explanation, show=False, max_display=SHAP_MAX_DISPLAY)
    plt.title("Feature Impact on This Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    return save_path


def main():
    """Run the full evaluation pipeline and export plots for the app/report."""
    logger.info("=" * 60)
    logger.info("PEDIATRIC APPENDICITIS — MODEL EVALUATION")
    logger.info("=" * 60)
    # Load model and re-run data preparation for a consistent test split.
    os.makedirs(IMAGES_DIR, exist_ok=True)

                         
    model, scaler, feature_names = load_model()
    df = load_data()
    df = optimize_memory(df)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df)
    logger.info("Evaluation pipeline data ready | X_train=%s | X_test=%s", X_train.shape, X_test.shape)

    # Compute metrics.
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

    # Export plots.
    plot_confusion_matrix(y_test, y_pred, os.path.join(IMAGES_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_prob, os.path.join(IMAGES_DIR, "roc_curve.png"))
    generate_shap_plots(model, X_test, feature_names, IMAGES_DIR)

    logger.info("All evaluation plots saved to %s", IMAGES_DIR)


if __name__ == "__main__":
    main()


