
"""Entraînement, comparaison, sélection et sauvegarde des modèles de classification."""

import os
import sys
import json
import logging
import warnings
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from src.config import MODELS_DIR, RANDOM_STATE, CV_FOLDS
except ImportError:
    from config import MODELS_DIR, RANDOM_STATE, CV_FOLDS


                                   

def get_models() -> dict:
    """Return the candidate model set with tuned baseline hyperparameters."""
    return {
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            C=1.0,
            gamma="scale",
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight=None,                                                
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,                                         
            num_leaves=31,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=10,                                               
            learning_rate=0.01,                                           
            random_seed=RANDOM_STATE,
            auto_class_weights="Balanced",
            verbose=0,
        ),
    }


                                       

def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """Train each model, run CV on train split, and compute test metrics."""
    logger.info(
        "Train/eval start | X_train=%s | X_test=%s | positives_train=%.3f",
        X_train.shape,
        X_test.shape,
        float(np.mean(y_train)),
    )
    models  = get_models()
    results = {}

    for name, model in models.items():
        logger.info("Entraînement : %s...", name)

        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS,
                                    scoring="roc_auc", n_jobs=-1)
        logger.info("  CV ROC-AUC : %.4f (± %.4f)",
                    cv_scores.mean(), cv_scores.std())

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":        round(accuracy_score(y_test, y_pred), 4),
            "precision":       round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":          round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":         round(roc_auc_score(y_test, y_prob), 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std":  round(cv_scores.std(), 4),
        }

        results[name] = {"model": model, "metrics": metrics}
        logger.info(
            "  Test → Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
            metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["f1_score"], metrics["roc_auc"],
        )

    return results


                                         

def select_best_model(results: dict) -> tuple:
    """Select the best model using recall, then precision, then ROC-AUC."""
    best_name = max(
        results,
        key=lambda k: (
            results[k]["metrics"]["recall"],
            results[k]["metrics"]["precision"],
            results[k]["metrics"]["roc_auc"],
        )
    )
    best = results[best_name]
    logger.info(
        "Meilleur modèle : %s | Recall=%.4f | Prec=%.4f | AUC=%.4f",
        best_name,
        best["metrics"]["recall"],
        best["metrics"]["precision"],
        best["metrics"]["roc_auc"],
    )
    return best_name, best["model"], best["metrics"]


                                     

def save_artifacts(model, scaler, feature_names: list,
                   best_name: str, all_results: dict):
    """Persist model artifacts and metrics summary in the models directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model,         os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler,        os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    metrics_summary = {
        "best_model": best_name,
        "models": {name: data["metrics"] for name, data in all_results.items()},
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Artefacts sauvegardés dans %s", MODELS_DIR)
    logger.debug("Saved artifacts | best_model=%s | n_features=%d", best_name, len(feature_names))


                               

def main(X_train, X_test, y_train, y_test, scaler, feature_names):
    """Execute the training pipeline and return all model results + best name."""
    logger.info("=" * 65)
    logger.info("PEDIATRIC APPENDICITIS — TRAINING PIPELINE")
    logger.info("=" * 65)
    logger.info("Pipeline input | features=%d", len(feature_names))

    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_name, best_model, best_metrics = select_best_model(results)

    # Final report for the selected champion model.
    y_pred_final = best_model.predict(X_test)
    print("\n" + "=" * 65)
    print(f"RAPPORT FINAL — {best_name}")
    print("=" * 65)
    print(classification_report(y_test, y_pred_final,
                                 target_names=["Pas appendicite", "Appendicite"]))

    # Side-by-side metrics table for all candidates.
    print("=" * 80)
    print(f"{'Modèle':<18} {'Accuracy':>10} {'Précision':>10} "
          f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 80)
    for name, data in results.items():
        m      = data["metrics"]
        marker = " ★" if name == best_name else ""
        print(
            f"{name + marker:<18} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['roc_auc']:>10.4f}"
        )
    print("=" * 80)
    print(f"\n✓ Meilleur modèle : {best_name}\n")

    save_artifacts(best_model, scaler, feature_names, best_name, results)

    return results, best_name


if __name__ == "__main__":
    # Quick local run when called directly from this module.
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_processing import load_data, optimize_memory, clean_data, preprocess_data

    df = load_data()
    df = optimize_memory(df)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    main(X_train, X_test, y_train, y_test, scaler, feature_names)