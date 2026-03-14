"""Recherche d'hyperparamètres par CV puis validation finale des champions."""

import os
import sys
import json
import itertools
import logging
import warnings
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_processing import load_data, optimize_memory, clean_data, preprocess_data

try:
    from src.config import MODELS_DIR, RESULTS_FILE, RANDOM_STATE, CV_FOLDS
except ImportError:
    from config import MODELS_DIR, RESULTS_FILE, RANDOM_STATE, CV_FOLDS

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Shared CV splitter for all models to keep comparisons fair.
CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Parameter grids centered around values that worked in prior runs.
GRIDS = {
    "SVM": {
        "C":            [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "gamma":        ["scale", "auto", 0.01, 0.1],
        "class_weight": ["balanced", None],
    },
    "Random Forest": {
        "n_estimators":      [100, 150, 200, 300],
        "max_depth":         [7, 8, 10, 12, 15, None],
        "min_samples_split": [2, 3, 5, 8],
        "min_samples_leaf":  [1, 2, 3],
        "class_weight":      [None],                                                 
    },
    "LightGBM": {
        "n_estimators":  [100, 150, 200, 300],
        "max_depth":     [4, 5, 6, 7, 8],
        "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "num_leaves":    [20, 31, 40, 50],
        "class_weight":  ["balanced", None],
    },
    "CatBoost": {
        "iterations":         [100, 150, 200, 300, 400],
        "depth":              [5, 6, 7, 8, 10],
        "learning_rate":      [0.01, 0.03, 0.05, 0.08, 0.1],
        "auto_class_weights": ["Balanced", None],
    },
}


def build_model(name: str, params: dict):
    """Build one model instance from its name and parameter dictionary."""
    logger.debug("Building model | name=%s | params=%s", name, params)
    if name == "SVM":
        return SVC(**params, kernel="rbf", probability=True, random_state=RANDOM_STATE)
    elif name == "Random Forest":
        return RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == "LightGBM":
        return LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    elif name == "CatBoost":
        return CatBoostClassifier(**params, random_seed=RANDOM_STATE, verbose=0)
    raise ValueError(f"Unknown model name: {name}")


def cv_score(model, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Evaluate one model configuration with stratified cross-validation."""
    logger.debug("CV scoring | model=%s | samples=%d", type(model).__name__, len(X_train))
    scoring = {
        "recall":    "recall",
        "precision": "precision",
        "roc_auc":   "roc_auc",
    }
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=CV,
        scoring=scoring,
        n_jobs=-1,
        error_score=0.0,
    )
    return {
        "recall":    round(cv_results["test_recall"].mean(), 4),
        "precision": round(cv_results["test_precision"].mean(), 4),
        "roc_auc":   round(cv_results["test_roc_auc"].mean(), 4),
                                        # Keep variability to detect unstable parameter sets.
        "recall_std":    round(cv_results["test_recall"].std(), 4),
        "precision_std": round(cv_results["test_precision"].std(), 4),
        "roc_auc_std":   round(cv_results["test_roc_auc"].std(), 4),
    }


def final_test_score(model, X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Fit on train split and compute final metrics on held-out test split."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }


def grid_combinations(grid: dict) -> list:
    """Expand a parameter grid dictionary into full Cartesian combinations."""
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def tune_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Run parameter search by CV and keep best + top3 combos per model."""
    logger.info("Tuning started | train_shape=%s", X_train.shape)
    tuning_results = {}

    for model_name, grid in GRIDS.items():
        combos = grid_combinations(grid)
        logger.info(
            "TUNING %-16s — %d combinaisons × 5 folds CV...",
            model_name, len(combos)
        )

        scores = []
        for i, params in enumerate(combos, 1):
            if i % 500 == 0:
                logger.info("  %s : %d / %d combinaisons...", model_name, i, len(combos))
            try:
                model   = build_model(model_name, params)
                metrics = cv_score(model, X_train, y_train)
                scores.append({"params": params, "cv_metrics": metrics})
            except Exception as e:
                logger.warning("Combinaison ignorée (%s) : %s", params, e)

                                # Lexicographic ranking: recall first, then precision, then AUC.
        scores.sort(
            key=lambda x: (
                x["cv_metrics"]["recall"],
                x["cv_metrics"]["precision"],
                x["cv_metrics"]["roc_auc"],
            ),
            reverse=True,
        )

        best = scores[0]
        top3 = scores[:3]
        tuning_results[model_name] = {"best": best, "top3": top3}

        logger.info(
            "  ★ Meilleurs params CV → Recall=%.4f (±%.4f) | "
            "Prec=%.4f (±%.4f) | AUC=%.4f (±%.4f)",
            best["cv_metrics"]["recall"],    best["cv_metrics"]["recall_std"],
            best["cv_metrics"]["precision"], best["cv_metrics"]["precision_std"],
            best["cv_metrics"]["roc_auc"],   best["cv_metrics"]["roc_auc_std"],
        )
        logger.info("  Params : %s", best["params"])

    return tuning_results


def evaluate_champions(tuning_results: dict,
                        X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate each model's best CV params once on the test split."""
    logger.info("=" * 65)
    logger.info("ÉVALUATION FINALE SUR X_TEST (une seule fois par modèle)")
    logger.info("=" * 65)

    final_results = {}

    for model_name, data in tuning_results.items():
        best_params  = data["best"]["params"]
        cv_metrics   = data["best"]["cv_metrics"]
        logger.info("Evaluating champion | model=%s | params=%s", model_name, best_params)

        model        = build_model(model_name, best_params)
        test_metrics = final_test_score(model, X_train, X_test, y_train, y_test)

        final_results[model_name] = {
            "best_params":  best_params,
            "cv_metrics":   cv_metrics,
            "test_metrics": test_metrics,
        }

        logger.info(
            "%-16s → Test Recall=%.4f | Prec=%.4f | AUC=%.4f  "
            "(CV Recall=%.4f ± %.4f)",
            model_name,
            test_metrics["recall"], test_metrics["precision"], test_metrics["roc_auc"],
            cv_metrics["recall"],   cv_metrics["recall_std"],
        )

    return final_results


def print_report(tuning_results: dict, final_results: dict):
    """Print a readable console summary for CV and final test performance."""
    # CV summary by model.
    print("\n" + "=" * 80)
    print("PHASE 1 — MEILLEURS HYPERPARAMÈTRES PAR CROSS-VALIDATION")
    print("=" * 80)

    for model_name, data in tuning_results.items():
        best = data["best"]
        cv   = best["cv_metrics"]

        print(f"\n{'─' * 80}")
        print(f"  {model_name}")
        print(f"{'─' * 80}")
        print(f"  Scores CV moyens (5-fold sur X_train) :")
        print(f"    Recall    : {cv['recall']:.4f}  (± {cv['recall_std']:.4f})")
        print(f"    Précision : {cv['precision']:.4f}  (± {cv['precision_std']:.4f})")
        print(f"    ROC-AUC   : {cv['roc_auc']:.4f}  (± {cv['roc_auc_std']:.4f})")
        print(f"  Hyperparamètres retenus :")
        for k, v in best["params"].items():
            print(f"    {k:<22} = {v}")

        print(f"\n  Top 3 (CV) :")
        print(f"  {'Recall':>8} {'±':>6} {'Précision':>10} {'±':>6} "
              f"{'AUC':>8} {'±':>6}   Diff vs best")
        print(f"  {'─' * 70}")
        for entry in data["top3"]:
            m    = entry["cv_metrics"]
            diff = {k: v for k, v in entry["params"].items()
                    if v != best["params"].get(k)}
            diff_str = ", ".join(f"{k}={v}" for k, v in diff.items()) or "identique"
            print(
                f"  {m['recall']:>8.4f} {m['recall_std']:>6.4f} "
                f"{m['precision']:>10.4f} {m['precision_std']:>6.4f} "
                f"{m['roc_auc']:>8.4f} {m['roc_auc_std']:>6.4f}   {diff_str}"
            )

    # Final head-to-head on test split.
    print("\n\n" + "=" * 80)
    print("PHASE 2 — COMPARAISON FINALE DES 4 CHAMPIONS SUR X_TEST")
    print("(X_test utilisé ici pour la première et unique fois)")
    print("=" * 80)
    print(f"\n  {'Modèle':<18} {'Recall':>10} {'Précision':>10} {'ROC-AUC':>10}")
    print(f"  {'─' * 52}")

    # Show best model first.
    sorted_models = sorted(
        final_results.items(),
        key=lambda x: (
            x[1]["test_metrics"]["recall"],
            x[1]["test_metrics"]["precision"],
            x[1]["test_metrics"]["roc_auc"],
        ),
        reverse=True,
    )

    for i, (model_name, data) in enumerate(sorted_models):
        t      = data["test_metrics"]
        marker = " ★" if i == 0 else ""
        print(
            f"  {model_name + marker:<18} "
            f"{t['recall']:>10.4f} {t['precision']:>10.4f} {t['roc_auc']:>10.4f}"
        )

    best_model_name = sorted_models[0][0]
    print(f"\n  ✓ Meilleur modèle (test) : {best_model_name}")
    print("=" * 80)


def main():
    """Run tuning pipeline and save results in JSON format."""
    logger.info("=" * 65)
    logger.info("PARAMETER SEARCH — PEDIATRIC APPENDICITIS")
    logger.info("=" * 65)

    # Prepare train/test arrays with the same preprocessing as production.
    df = load_data()
    df = optimize_memory(df)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    # Phase 1: hyperparameter search on train split.
    tuning_results = tune_all_models(X_train, y_train)

    # Phase 2: evaluate champion of each model family on test split.
    final_results = evaluate_champions(tuning_results, X_train, X_test, y_train, y_test)

    # Human-readable console report.
    print_report(tuning_results, final_results)

    # Persist full report for later analysis.
    os.makedirs(MODELS_DIR, exist_ok=True)
    output = {
        "tuning":  tuning_results,
        "final":   final_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Résultats sauvegardés → %s", RESULTS_FILE)

    return tuning_results, final_results


if __name__ == "__main__":
    main()