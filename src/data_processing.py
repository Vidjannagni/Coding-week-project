"""Préparation des données: chargement, nettoyage, feature engineering et prétraitement."""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure imports work both as package and as direct script execution.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from src.config import (
        DATA_DIR, TARGET_COL, TARGET_COLS, LEAKAGE_COLS, CIRCULAR_COLS, WEAK_COLS,
        MISSING_THRESHOLD, CORR_THRESHOLD, IQR_FACTOR, MIN_CORR_TRAIN_SIZE,
        HEIGHT_CM_THRESHOLD, CATEGORY_RATIO_THRESHOLD, RANDOM_STATE, TEST_SIZE,
    )
except ImportError:
    from config import (
        DATA_DIR, TARGET_COL, TARGET_COLS, LEAKAGE_COLS, CIRCULAR_COLS, WEAK_COLS,
        MISSING_THRESHOLD, CORR_THRESHOLD, IQR_FACTOR, MIN_CORR_TRAIN_SIZE,
        HEIGHT_CM_THRESHOLD, CATEGORY_RATIO_THRESHOLD, RANDOM_STATE, TEST_SIZE,
    )


                       

def load_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load the dataset locally, or download and cache it on first run."""
    csv_path = os.path.join(data_dir, "appendicitis.csv")

    if os.path.exists(csv_path):
        logger.info("Chargement depuis le fichier local : %s", csv_path)
        df = pd.read_csv(csv_path)
    else:
        logger.info("Téléchargement depuis UCI ML Repository...")
        os.makedirs(data_dir, exist_ok=True)
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=938)
            X = dataset.data.features
            y = dataset.data.targets
            if isinstance(y, pd.DataFrame):
                target = y["Diagnosis"] if "Diagnosis" in y.columns else y.iloc[:, 0]
            else:
                target = y
            df = pd.concat([X, target.rename("Diagnosis")], axis=1)
            df.to_csv(csv_path, index=False)
            logger.info("Dataset sauvegardé : %s (%d lignes, %d colonnes)",
                        csv_path, len(df), len(df.columns))
        except Exception as e:
            logger.error("Échec du téléchargement : %s", e)
            raise

    logger.info("Dataset ready | rows=%d | cols=%d", len(df), len(df.columns))
    return df


                                 

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce dataframe memory footprint by downcasting numeric/object columns."""
    mem_before = df.memory_usage(deep=True).sum() / 1024**2
    logger.info("Mémoire AVANT : %.2f MB", mem_before)
    df_opt = df.copy()

    for col in df_opt.select_dtypes(include=["float64"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")
    for col in df_opt.select_dtypes(include=["int64", "int32", "int16"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")
    for col in df_opt.select_dtypes(include=["object"]).columns:
        if df_opt[col].nunique() / len(df_opt) < CATEGORY_RATIO_THRESHOLD:
            df_opt[col] = df_opt[col].astype("category")

    mem_after = df_opt.memory_usage(deep=True).sum() / 1024**2
    logger.info("Mémoire APRÈS : %.2f MB (réduction %.1f%%)",
                mem_after, (1 - mem_after / mem_before) * 100)
    logger.debug("Dtypes after optimize_memory: %s", df_opt.dtypes.value_counts().to_dict())
    return df_opt


                                 

def _impute_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct missing BMI from weight/height when possible, then drop both."""
       
    df = df.copy()

    # Case-insensitive column lookup to support naming variants.
    col_map    = {c.lower(): c for c in df.columns}
    bmi_col    = col_map.get("bmi")
    weight_col = col_map.get("weight") or col_map.get("weight_kg")
    height_col = (col_map.get("height") or col_map.get("height_cm")
                  or col_map.get("height_m"))

    if bmi_col is None:
        logger.info("Colonne BMI non trouvée — étape BMI ignorée.")
        return df

    if weight_col is None or height_col is None:
        logger.info("Colonnes poids/taille non trouvées — BMI non recalculable par formule.")
        return df

    # If median height is > threshold, values are likely in centimeters.
    height_median = df[height_col].median()
    height_in_cm  = height_median > HEIGHT_CM_THRESHOLD
    height_m      = df[height_col] / 100.0 if height_in_cm else df[height_col]
    if height_in_cm:
        logger.info("Taille détectée en cm (médiane=%.1f) — conversion en mètres.", height_median)

    # Compute BMI only where source values are valid and BMI is missing.
    mask_missing  = df[bmi_col].isna()
    mask_has_data = df[weight_col].notna() & df[height_col].notna() & (height_m > 0)
    mask_calc     = mask_missing & mask_has_data

    n_calc = mask_calc.sum()
    if n_calc > 0:
        df.loc[mask_calc, bmi_col] = (
            df.loc[mask_calc, weight_col] / (height_m[mask_calc] ** 2)
        )
        logger.info("BMI calculé par formule pour %d lignes.", n_calc)

    n_still = df[bmi_col].isna().sum()
    if n_still > 0:
        logger.info("%d BMI toujours manquants → traités à l'étape suivante.", n_still)

    # Drop raw height/weight to avoid deterministic redundancy with BMI.
    cols_to_drop = [c for c in [weight_col, height_col] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Colonnes supprimées après reconstruction BMI : %s", cols_to_drop)
    return df


def _impute_by_correlation(df: pd.DataFrame,
                            corr_threshold: float = CORR_THRESHOLD,
                            target_col: str = TARGET_COL) -> pd.DataFrame:
    """Impute missing numeric values using linear regression on correlated pairs."""
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_col]

    # Correlation matrix over numeric features only.
    corr_matrix = df[num_cols].corr().abs()

    # Use upper triangle pairs only to avoid duplicates.
    pairs = [
        (col_a, col_b, corr_matrix.loc[col_a, col_b])
        for i, col_a in enumerate(num_cols)
        for col_b in num_cols[i+1:]
        if corr_matrix.loc[col_a, col_b] >= corr_threshold
    ]

    if not pairs:
        logger.info("Aucune paire avec |r| ≥ %.2f — imputation par corrélation ignorée.",
                    corr_threshold)
        return df

    logger.info("%d paire(s) identifiée(s) pour imputation (|r| ≥ %.2f).",
                len(pairs), corr_threshold)

    total_imputed = 0
    # Strongest correlations first to maximize useful imputations early.
    for col_a, col_b, r in sorted(pairs, key=lambda x: -x[2]):
        for target_var, predictor_var in [(col_a, col_b), (col_b, col_a)]:
            nan_target    = df[target_var].isna()
            nan_predictor = df[predictor_var].isna()

            # Predict rows where target is missing but predictor exists.
            mask_predict = nan_target & ~nan_predictor
            if mask_predict.sum() == 0:
                continue

            # Fit rows where both variables are available.
            train_mask = ~nan_target & ~nan_predictor
            if train_mask.sum() < MIN_CORR_TRAIN_SIZE:
                # Skip under-sampled regressions.
                continue

            reg = LinearRegression()
            reg.fit(df.loc[train_mask, [predictor_var]],
                    df.loc[train_mask, target_var])
            df.loc[mask_predict, target_var] = reg.predict(
                df.loc[mask_predict, [predictor_var]]
            )
            total_imputed += mask_predict.sum()
            logger.info("  %s ← %s (r=%.3f) : %d valeurs imputées.",
                        target_var, predictor_var, r, mask_predict.sum())

            # Masks are recomputed each loop, so newly imputed values can help later pairs.

    logger.info("Imputation par corrélation : %d valeurs imputées.", total_imputed)
    return df


def _drop_correlated_features(df: pd.DataFrame,
                               corr_threshold: float = CORR_THRESHOLD,
                               target_col: str = TARGET_COL,
                               original_nan_counts: pd.Series = None) -> tuple:
    """Drop redundant numeric features while preserving target-correlated signals."""
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_col]

    # Build a temporary numeric target to score feature usefulness.
    target_numeric = df[target_col].copy()
    if not pd.api.types.is_numeric_dtype(target_numeric):
        target_numeric = target_numeric.astype(str).str.strip().str.lower()
        target_numeric = target_numeric.map(
            lambda v: 0 if "no" in v else (1 if "appendicitis" in v else np.nan)
        )

    # Compute target correlations using corrwith to avoid embedding a
    # FloatingArray (pandas 2.3 extension type) into df_for_corr, which
    # would cause "underlying array is read-only" when corr() calls
    # to_numpy(copy=False) on numpy 2.x.
    df_for_corr = df[num_cols].copy()
    numeric_target = np.array(pd.to_numeric(target_numeric, errors="coerce"),
                              dtype=np.float64)
    # Greedy loop: drop one redundant column at a time.
    target_corr = df_for_corr.corrwith(
        pd.Series(numeric_target, index=df_for_corr.index)
    ).abs()

    dropped = []

                                                                                   
    while True:
                                    # Recompute on remaining columns after each drop.
        remaining = [c for c in num_cols if c not in dropped]
        if len(remaining) < 2:
            break

        corr_matrix = df[remaining].copy().corr().abs()
        # Rebuild from a writable numpy copy — pandas 2.3 CoW makes
        # .values read-only, which breaks np.fill_diagonal in-place.
        corr_values = corr_matrix.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(corr_values, 0)
        corr_matrix = pd.DataFrame(corr_values,
                                   index=corr_matrix.index,
                                   columns=corr_matrix.columns)
        redundancy_score = (corr_matrix >= corr_threshold).sum()

        if redundancy_score.max() == 0:
            # No redundant pair remains.
            break

                            # Keep columns with stronger target link; drop weaker ones first.
        candidates = redundancy_score[redundancy_score > 0]

                        # Tie-break with original missingness, then stable alphabetical order.
        col_to_drop = min(
            candidates.index,
            key=lambda c: (
                                target_corr.get(c, 0),
                                -(original_nan_counts.get(c, 0)
                  if original_nan_counts is not None else 0),
                                c
            )
        )

                # Keep traceability in logs about why a column is removed.
        conflicting = corr_matrix[col_to_drop][corr_matrix[col_to_drop] >= corr_threshold].index.tolist()
        logger.info(
            "Étape 9 — Suppression '%s' (r ≥ %.2f avec : %s | corr_cible=%.3f)",
            col_to_drop, corr_threshold, conflicting,
            target_corr.get(col_to_drop, 0)
        )
        dropped.append(col_to_drop)

    if dropped:
        df = df.drop(columns=dropped)
        logger.info(
            "Étape 9 — %d colonne(s) supprimées pour redondance : %s",
            len(dropped), dropped
        )
    else:
        logger.info("Étape 9 — Aucune colonne redondante détectée (seuil %.2f).", corr_threshold)

    return df, dropped


                                

def clean_data(df: pd.DataFrame,
               missing_threshold: float = MISSING_THRESHOLD,
               corr_threshold: float = CORR_THRESHOLD) -> pd.DataFrame:
    """Apply the full cleaning pipeline used before model training."""
    logger.info("Nettoyage : %d lignes, %d colonnes.", *df.shape)
    df_clean   = df.copy()
    target_col = TARGET_COL

    # Step 1: remove leak-prone, circular or weak predictors.
                                                            
     
                                                               
                                
     
                                           
                                                                     
     
                                                                          
                                                                                   
                                                         
     
                                                                        
                                                                              
                                    
                                    
                                                                        
                                                                          
                                                                                 
                                                                                    
                                                                                               
    cols_to_drop = [c for c in TARGET_COLS + LEAKAGE_COLS + CIRCULAR_COLS + WEAK_COLS
                    if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        logger.info("Étape 1 — %d colonne(s) supprimées : %s", len(cols_to_drop), cols_to_drop)

    # Step 2: drop very sparse columns.
    feature_cols = [c for c in df_clean.columns if c != target_col]
    missing_rate = df_clean[feature_cols].isnull().mean()
    high_missing = missing_rate[missing_rate > missing_threshold].index.tolist()
    if high_missing:
        logger.info(
            "Étape 2 — %d colonne(s) supprimées (NaN > %.0f%%) :\n%s",
            len(high_missing), missing_threshold * 100,
            missing_rate[high_missing].sort_values(ascending=False).to_string()
        )
        df_clean = df_clean.drop(columns=high_missing)
    else:
        logger.info("Étape 2 — Aucune colonne > %.0f%% NaN.", missing_threshold * 100)

    # Save NaN counts before imputation for later tie-breaks.
    original_nan_counts = df_clean.isnull().sum()

    # Step 3: reconstruct BMI.
    logger.info("Étape 3 — Reconstruction BMI...")
    df_clean = _impute_bmi(df_clean)

    # Step 4: correlation-based imputation.
    logger.info("Étape 4 — Imputation par corrélation (|r| ≥ %.2f)...", corr_threshold)
    df_clean = _impute_by_correlation(df_clean, corr_threshold, target_col)

    # Step 5: derived clinical features.
    if "WBC_Count" in df_clean.columns and "CRP" in df_clean.columns:
        df_clean["WBC_CRP_Ratio"] = df_clean["WBC_Count"] / (df_clean["CRP"] + 0.1)
        logger.info("Étape 5 — Feature 'WBC_CRP_Ratio' créée.")

    if "Neutrophil_Percentage" in df_clean.columns and "WBC_Count" in df_clean.columns:
        df_clean["Neutrophil_WBC_Interaction"] = (
            df_clean["Neutrophil_Percentage"] * df_clean["WBC_Count"]
        )
        logger.info("Étape 5 — Feature 'Neutrophil_WBC_Interaction' créée.")

    # Step 6: residual median/mode imputation.
    num_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns
                if c != target_col]
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

    nan_num = df_clean[num_cols].isnull().sum().sum()
    nan_cat = df_clean[cat_cols].isnull().sum().sum() if cat_cols else 0

    if nan_num > 0:
        logger.info("Étape 6 — %d NaN numériques résiduels → médiane.", nan_num)
        df_clean[num_cols] = SimpleImputer(strategy="median").fit_transform(
            df_clean[num_cols])
    if nan_cat > 0:
        logger.info("Étape 6 — %d NaN catégoriels résiduels → mode.", nan_cat)
        df_clean[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
            df_clean[cat_cols])
    if nan_num == 0 and nan_cat == 0:
        logger.info("Étape 6 — Aucun NaN résiduel.")

    # Step 7: remove duplicate rows.
    n_dupes = df_clean.duplicated().sum()
    if n_dupes > 0:
        df_clean = df_clean.drop_duplicates()
        logger.info("Étape 7 — %d doublon(s) supprimé(s).", n_dupes)

    # Step 8: clip outliers with IQR fences.
    numeric_features = [c for c in df_clean.select_dtypes(include=[np.number]).columns
                        if c != target_col]
    total_capped = 0
    for col in numeric_features:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR    = Q3 - Q1
        lower, upper = Q1 - IQR_FACTOR * IQR, Q3 + IQR_FACTOR * IQR
        n_out  = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
        if n_out > 0:
            df_clean[col]  = df_clean[col].clip(lower, upper)
            total_capped  += n_out
    if total_capped > 0:
        logger.info("Étape 8 — %d valeurs cappées (IQR).", total_capped)

    # Step 9: prune redundant correlated features.
    logger.info("Étape 9 — Suppression colonnes redondantes (|r| ≥ %.2f)...", corr_threshold)
    df_clean, _ = _drop_correlated_features(
        df_clean,
        corr_threshold=corr_threshold,
        target_col=target_col,
        original_nan_counts=original_nan_counts,
    )

    logger.info("Nettoyage terminé : %d lignes, %d colonnes.", *df_clean.shape)
    return df_clean


                             

def preprocess_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Encode, split, and scale cleaned data for model training/evaluation."""
    logger.info("Prétraitement ML...")
    df_proc = df.copy()

    # Encode target robustly even when it is categorical text.
    if df_proc[target_col].dtype.name == "category":
        df_proc[target_col] = df_proc[target_col].astype(str)

    if df_proc[target_col].dtype == object or isinstance(df_proc[target_col].iloc[0], str):
        unique_vals = df_proc[target_col].unique()
        mapping = {}
        for val in unique_vals:
            val_str = str(val).strip().lower()
            if "no" in val_str:
                mapping[val] = 0
            elif "appendicitis" in val_str:
                mapping[val] = 1
            else:
                try:
                    mapping[val] = int(val)
                except (ValueError, TypeError):
                    mapping[val] = 0
        df_proc[target_col] = df_proc[target_col].map(mapping)
        logger.info("Mapping cible : %s", mapping)

    df_proc = df_proc.dropna(subset=[target_col])
    y = df_proc[target_col].astype(int)
    X = df_proc.drop(columns=[target_col])

    # One-hot encode categorical predictors.
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        logger.info("One-hot encoding : %s", cat_cols)

    feature_names = X.columns.tolist()
    logger.info("Preprocess feature space | n_features=%d", len(feature_names))

    # Stratified split before scaling to avoid data leakage.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Fit scaler on train only, then transform both splits.
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    logger.info(
        "Train : %d | Test : %d | Taux positifs — train : %.1f%%, test : %.1f%%",
        len(X_train), len(X_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )
    logger.debug("Preprocess output shapes | X_train=%s | X_test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test, scaler, feature_names


def get_class_distribution(y: pd.Series) -> dict:
    """Return class counts and percentages for quick diagnostics."""
    counts = y.value_counts()
    return {
        "counts":      counts.to_dict(),
        "percentages": (counts / len(y) * 100).round(1).to_dict(),
    }


                     


if __name__ == "__main__":
    df = load_data()
    print(f"\nShape brut : {df.shape}")
    df_opt   = optimize_memory(df)
    df_clean = clean_data(df_opt)
    print(f"Shape après nettoyage : {df_clean.shape}")
    print(f"Colonnes : {list(df_clean.columns)}")
    print(f"NaN résiduels : {df_clean.isnull().sum().sum()}")
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df_clean)
    print(f"\nFeatures ({len(features)}) : {features}")
    print(f"Train : {X_train.shape} | Test : {X_test.shape}")
    print(f"Distribution : {get_class_distribution(pd.Series(y_train))}")