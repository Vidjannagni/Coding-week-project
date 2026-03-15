# PediAppend -- AI-Powered Pediatric Appendicitis Decision Support

**Explainable machine-learning clinical decision support for pediatric appendicitis diagnosis**

[![CI Pipeline](https://github.com/Mavens06/Coding-week-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Mavens06/Coding-week-project/actions)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-1.2.4-orange.svg)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Architecture / Pipeline Diagram](#architecture--pipeline-diagram)
- [Dataset](#dataset)
- [Modeling](#modeling)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Web Application](#web-application)
- [Testing](#testing)
- [Limitations & Future Work](#limitations--future-work)
- [Prompt Engineering](#prompt-engineering)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

```bash
git clone https://github.com/Mavens06/Coding-week-project.git
cd Coding-week-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train the model (or skip -- pre-trained artifacts are in models/)
python src/train_model.py

# Launch the web app
cd app && python app.py
```

The app is available at **http://localhost:5000**.

| | |
|--|--|
| Default admin username | `admin` |
| Default admin password | `admin123` |

> **Documentation :** le journal de prompt engineering et les details de la methode de travail avec les agents IA sont dans [`docs/prompt_engineering.md`](docs/prompt_engineering.md).

---

## Overview

PediAppend is a clinical decision-support system that assists pediatricians in diagnosing appendicitis in children. It combines a **Random Forest** classifier (selected for highest recall -- critical in medical contexts) with **SHAP explainability** to produce transparent, interpretable risk assessments.

Given a set of clinical observations (symptoms, lab results, ultrasound findings, demographics), the system produces:

- A **probability score** for appendicitis
- A **risk classification** (high / moderate / low) with clinical recommendations
- **SHAP-based explanations** showing which clinical factors drove the prediction
- A **prediction history** persisted per user

**Target users:** pediatricians, emergency physicians, clinical researchers, and ML engineers evaluating explainable medical AI.

---

## Architecture / Pipeline Diagram

```
                         PediAppend ML Pipeline
 ============================================================================

  UCI Repository (ID 938)           Local CSV
        |                               |
        +---------- load_data() --------+
                        |
                optimize_memory()          float64->float32, int64 downcast,
                        |                  low-cardinality -> category
                   clean_data()
                        |
         +--------------+--------------+--------------+
         |              |              |              |
    Drop leaky/     BMI recon-    Correlation-    Feature
    weak/circular   struction     based           engineering
    columns         from Ht/Wt   imputation       WBC_CRP_Ratio
         |              |              |           Neutrophil_WBC
         +--------------+--------------+--------------+
                        |
              Median/mode imputation
              Deduplication
              IQR outlier capping
              Correlated feature pruning
                        |
                 preprocess_data()
                        |
         +--------------+--------------+
         |                             |
    Target encoding              One-hot encoding
    (Diagnosis -> 0/1)           (categorical features)
         |                             |
         +-------- Stratified ---------+
                   train/test
                   split (80/20)
                        |
         +---------+----+----+---------+
         |         |         |         |
        SVM     Random    LightGBM  CatBoost
       (RBF)    Forest
         |         |         |         |
         +----+----+----+----+----+----+
              |              |
        5-fold CV       Test-set metrics
        (ROC-AUC)       (Acc, Prec, Rec, F1, AUC)
              |
       select_best_model()
       (max recall, then precision, then AUC)
              |
       save_artifacts()
       best_model.pkl, scaler.pkl,
       feature_names.pkl, metrics.json
              |
       evaluate_model.py
       Confusion matrix, ROC curve,
       SHAP summary + beeswarm + waterfall
              |
       Flask web app (app/app.py)
       Per-patient prediction + SHAP explanation
```

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [Regensburg Pediatric Appendicitis](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis) (UCI ML Repository, ID 938) |
| **Patients** | 782 |
| **Raw features** | 53 |
| **Target** | `Diagnosis` -- binary classification (appendicitis vs. no appendicitis) |
| **Class balance** | ~59% appendicitis, ~41% no appendicitis (approximately balanced) |

### Data Acquisition

The dataset is auto-downloaded from the UCI repository on first run via `ucimlrepo` and cached locally at `data/appendicitis.csv`.

### Preprocessing Pipeline (9 steps in `clean_data()`)

| Step | Operation | Details |
|------|-----------|---------|
| 1 | Drop problematic columns | Leak-prone (`Length_of_Stay`), circular (`Alvarado_Score`, `Paedriatic_Appendicitis_Score`), weak predictors (10 columns), secondary targets (`Management`, `Severity`) |
| 2 | Drop sparse columns | Features with >50% missing values removed |
| 3 | BMI reconstruction | Recompute missing BMI from Height/Weight, then drop Height and Weight to avoid redundancy |
| 4 | Correlation-based imputation | Linear regression imputation for correlated numeric pairs (\|r\| >= 0.69) |
| 5 | Feature engineering | `WBC_CRP_Ratio` = WBC / (CRP + 0.1); `Neutrophil_WBC_Interaction` = Neutrophil% x WBC |
| 6 | Residual imputation | Median for remaining numeric NaNs, mode for categorical NaNs |
| 7 | Deduplication | Remove exact duplicate rows |
| 8 | Outlier capping | IQR method (1.5x IQR) -- values clipped, not removed |
| 9 | Redundant feature pruning | Greedy removal of highly correlated features (\|r\| >= 0.69), preferring features with stronger target correlation |

After cleaning, the pipeline retains **21 features** (6 numeric + 2 engineered + 12 binary one-hot encoded + peritonitis categories).

### Feature Encoding (`preprocess_data()`)

- Target: text labels mapped to `{0, 1}`
- Categorical predictors: one-hot encoded (`pd.get_dummies`, `drop_first=True`)
- Stratified 80/20 train/test split
- `StandardScaler` fit on train set only (no data leakage)

---

## Modeling

Four classifier families are evaluated:

| Model | Key Hyperparameters (defaults) | Why Included |
|-------|-------------------------------|--------------|
| **SVM (RBF)** | `C=1.0`, `gamma=scale`, `class_weight=balanced` | Strong margin-based classifier for moderate-dimensional data |
| **Random Forest** | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=2` | Robust ensemble with native feature importance; compatible with SHAP `TreeExplainer` |
| **LightGBM** | `n_estimators=100`, `max_depth=6`, `learning_rate=0.05`, `num_leaves=31` | Efficient gradient boosting with histogram-based splits |
| **CatBoost** | `iterations=200`, `depth=10`, `learning_rate=0.01`, `auto_class_weights=Balanced` | Gradient boosting with native categorical handling and built-in regularization |

### Model Selection Strategy

The best model is selected by **lexicographic ordering**:

1. **Recall** (highest priority -- minimize missed appendicitis cases)
2. **Precision** (tie-breaker)
3. **ROC-AUC** (second tie-breaker)

This recall-first strategy reflects the clinical requirement that missing an appendicitis diagnosis is more costly than a false positive.

### SHAP Explainability

- **Tree models** (Random Forest, LightGBM, CatBoost): use `shap.TreeExplainer` for exact, fast SHAP values
- **Non-tree models** (SVM): fall back to `shap.KernelExplainer` with a background sample of 50 instances
- Global plots: summary bar + beeswarm (top 15 features)
- Per-prediction: waterfall plot showing individual feature contributions

---

## Training

### Entry Point

```bash
# Full pipeline (recommended): load -> clean -> preprocess -> train -> save
python src/run.py

# Train only (requires data already preprocessed):
python src/train_model.py
```

### What Happens

1. `load_data()` -- loads or downloads the CSV dataset
2. `optimize_memory()` -- downcasts dtypes (reduces memory ~93%)
3. `clean_data()` -- applies the 9-step cleaning pipeline
4. `preprocess_data()` -- encodes, splits (stratified 80/20), scales
5. `train_and_evaluate()` -- trains all 4 models, runs 5-fold CV on the train split, evaluates on test split
6. `select_best_model()` -- picks the champion by (recall, precision, AUC)
7. `save_artifacts()` -- persists model, scaler, feature names, and metrics

### Saved Artifacts (in `models/`)

| File | Contents |
|------|----------|
| `best_model.pkl` | The selected best model (currently Random Forest) |
| `scaler.pkl` | Fitted `StandardScaler` |
| `feature_names.pkl` | Ordered list of 21 feature names |
| `metrics.json` | Full metrics for all 4 models |
| `threshold.txt` | Optimal classification threshold (0.498) |
| `feature_selector.pkl` | Feature selector artifact |

---

## Evaluation

### Entry Point

```bash
python src/evaluate_model.py
```

This loads the saved model, re-prepares the test split (same random seed ensures identical split), and generates:

- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix plot (`reports/images/confusion_matrix.png`)
- ROC curve plot (`reports/images/roc_curve.png`)
- SHAP summary bar plot (`reports/images/shap_summary_bar.png`)
- SHAP beeswarm plot (`reports/images/shap_beeswarm.png`)
- SHAP waterfall plot for sample 0 (`reports/images/shap_waterfall.png`)

### Metrics (from `models/metrics.json`)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | CV ROC-AUC |
|-------|----------|-----------|--------|----------|---------|------------|
| SVM | 0.8599 | 0.9277 | 0.8280 | 0.8750 | 0.9524 | 0.9349 +/- 0.0255 |
| **Random Forest** | **0.9363** | **0.9192** | **0.9785** | **0.9479** | **0.9756** | 0.9491 +/- 0.0268 |
| LightGBM | 0.9236 | 0.9263 | 0.9462 | 0.9362 | 0.9686 | 0.9587 +/- 0.0211 |
| CatBoost | 0.9172 | 0.9255 | 0.9355 | 0.9305 | 0.9783 | 0.9613 +/- 0.0214 |

**Selected model: Random Forest** -- highest recall (97.85%), which is the primary selection criterion for this medical application.

---

## Hyperparameter Tuning

### Strategy

Full Cartesian grid search with **stratified 5-fold cross-validation** on the training split. Each model family's champion is then evaluated once on the held-out test split.

**Entry point:**

```bash
python src/tuning.py
```

### Search Spaces

| Model | Parameter | Values Searched |
|-------|-----------|----------------|
| **SVM** | `C` | 0.1, 0.5, 1.0, 2.0, 5.0, 10.0 |
| | `gamma` | scale, auto, 0.01, 0.1 |
| | `class_weight` | balanced, None |
| **Random Forest** | `n_estimators` | 100, 150, 200, 300 |
| | `max_depth` | 7, 8, 10, 12, 15, None |
| | `min_samples_split` | 2, 3, 5, 8 |
| | `min_samples_leaf` | 1, 2, 3 |
| | `class_weight` | None |
| **LightGBM** | `n_estimators` | 100, 150, 200, 300 |
| | `max_depth` | 4, 5, 6, 7, 8 |
| | `learning_rate` | 0.03, 0.05, 0.08, 0.1, 0.15 |
| | `num_leaves` | 20, 31, 40, 50 |
| | `class_weight` | balanced, None |
| **CatBoost** | `iterations` | 100, 150, 200, 300, 400 |
| | `depth` | 5, 6, 7, 8, 10 |
| | `learning_rate` | 0.01, 0.03, 0.05, 0.08, 0.1 |
| | `auto_class_weights` | Balanced, None |

### Best Parameters Found (from `models/param_search_results.json`)

| Model | Best CV Params | CV Recall | CV Precision | CV ROC-AUC |
|-------|---------------|-----------|--------------|------------|
| SVM | C=0.1, gamma=0.1, class_weight=None | 0.9758 | 0.6529 | 0.8975 |
| Random Forest | n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=2 | 0.9274 | 0.9277 | 0.9566 |
| LightGBM | n_estimators=300, max_depth=7, lr=0.15, num_leaves=31 | 0.9139 | 0.9244 | 0.9577 |
| CatBoost | iterations=200, depth=6, lr=0.03 | 0.9274 | 0.9181 | 0.9655 |

### Final Test-Set Performance (champions only)

| Model | Test Recall | Test Precision | Test ROC-AUC |
|-------|-------------|----------------|--------------|
| SVM | 1.0000 | 0.6549 | 0.9094 |
| **Random Forest** | **0.9785** | **0.9100** | **0.9743** |
| LightGBM | 0.9462 | 0.8980 | 0.9614 |
| CatBoost | 0.9462 | 0.9072 | 0.9751 |

---

## Project Structure

```
Coding-week-project/
|
|-- .github/
|   `-- workflows/
|       `-- ci.yml                       # CI: imports, unit tests, data pipeline, Flask routes, ML artifacts, release
|
|-- app/                                 # Flask web application
|   |-- app.py                           # Main app: routes (/,  /diagnosis, /predict), feature vector builder
|   |-- auth.py                          # Auth blueprint: register, login, logout, profile, history, admin
|   |-- config.py                        # App constants: paths, feature maps, auth defaults
|   |-- shap_utils.py                    # SHAP explainer init + per-prediction SHAP value computation
|   |-- history.db                       # SQLite database (users + prediction history)
|   |-- static/
|   |   |-- css/
|   |   |   |-- core.css                # Base layout, typography, utilities
|   |   |   |-- form.css               # Diagnosis form wizard styles
|   |   |   |-- landing.css            # Landing page hero and sections
|   |   |   |-- pages.css              # Login, register, history, admin, profile
|   |   |   |-- result.css             # Prediction result page
|   |   |   `-- style.css              # Global imports and shared variables
|   |   |-- img/
|   |   |   |-- anatomy-body.jpg
|   |   |   |-- appendix-anatomy.svg
|   |   |   `-- favicon.svg
|   |   `-- js/
|   |       |-- common.js                # Global UI (navbar, animations, flash)
|   |       |-- landing.js               # Landing page slides + stats + particles
|   |       |-- diagnosis.js             # Multi-step form wizard + validation + BMI/Age
|   |       |-- result.js                # Result ring + SHAP bars animations
|   |       |-- history.js               # History filtering + delete/clear actions
|   |       `-- admin.js                 # Admin toggle/delete actions
|   `-- templates/
|       |-- base.html                    # Base layout (navbar, footer, flash messages)
|       |-- index.html                   # Landing page: 3-slide hero, stats, features
|       |-- diagnosis.html               # 3-step form wizard (demographics, symptoms, lab/exam)
|       |-- result.html                  # Prediction result: ring chart, SHAP waterfall, anatomy ref
|       |-- login.html                   # Login form
|       |-- register.html                # Registration form
|       |-- profile.html                 # User profile editing
|       |-- history.html                 # Searchable/filterable prediction history
|       `-- admin.html                   # Admin dashboard: user management
|
|-- catboost_info/
|   `-- catboost_training.json           # CatBoost training metadata (auto-generated)
|
|-- data/
|   |-- appendicitis.csv                 # Raw dataset (782 patients, 53 features)
|   `-- processed/
|       |-- X_processed.csv              # Preprocessed feature matrix (from EDA notebook)
|       `-- y_processed.csv              # Preprocessed target labels (from EDA notebook)
|
|-- docs/
|   `-- prompt_engineering.md            # Prompt engineering journal: prompts, results, lessons learned
|
|-- models/
|   |-- best_model.pkl                   # Trained best model (Random Forest)
|   |-- scaler.pkl                       # Fitted StandardScaler
|   |-- feature_names.pkl                # Ordered list of 21 feature names
|   |-- feature_selector.pkl             # Feature selector artifact
|   |-- metrics.json                     # Per-model evaluation metrics
|   |-- param_search_results.json        # Full hyperparameter tuning results + top-3 per model
|   `-- threshold.txt                    # Optimal classification threshold (0.498)
|
|-- notebooks/
|   `-- eda.ipynb                        # Exploratory data analysis (41 cells): missing values,
|                                        #   class balance, outliers, correlations, memory optimization
|
|-- reports/
|   |-- figures/                         # EDA visualizations (generated by notebook)
|   |   |-- boxplots_by_diagnosis.png
|   |   |-- class_distribution.png
|   |   |-- correlation_matrix.png
|   |   |-- correlation_pairs.png
|   |   |-- distributions.png
|   |   |-- imputation_regression.png
|   |   |-- memory_optimization.png
|   |   `-- missing_values.png
|   `-- images/                          # Model evaluation plots (generated by evaluate_model.py)
|       |-- confusion_matrix.png
|       |-- roc_curve.png
|       |-- shap_summary_bar.png
|       `-- shap_beeswarm.png
|
|-- src/                                 # ML pipeline source code
|   |-- __init__.py                      # Package version: "1.2.4"
|   |-- config.py                        # Central config: paths, ML defaults, cleaning thresholds
|   |-- data_processing.py               # load, optimize, clean (9 steps), preprocess
|   |-- train_model.py                   # Train 4 models, CV, select best, save artifacts
|   |-- evaluate_model.py                # Metrics, confusion matrix, ROC curve, SHAP plots
|   |-- tuning.py                        # Cartesian grid search + champion evaluation
|   `-- run.py                           # End-to-end pipeline entry point
|
|-- tests/                               # Pytest test suite (~80 tests)
|   |-- __init__.py
|   |-- conftest.py                      # Session-scoped fixtures (loaded data, model artifacts)
|   |-- test_data_processing.py          # Data loading, memory optimization, cleaning, preprocessing
|   |-- test_train_model.py              # Training, model selection, saved model validation
|   |-- test_evaluate_model.py           # Evaluation metrics, plot generation
|   |-- test_app.py                      # Flask routes, auth flow, feature vector builder
|   |-- test_run.py                      # Pipeline integration + reproducibility
|   `-- test_tuning.py                   # Grid search, CV scoring, champion evaluation
|
|-- requirements.txt                     # Python dependencies
`-- README.md                            # This file
```

---

## Installation

### Prerequisites

- Python 3.11 or higher (CI runs on Python 3.13)
- pip
- (Optional) a virtual environment manager (`venv`, `conda`, etc.)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Mavens06/Coding-week-project.git
cd Coding-week-project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the training pipeline (downloads the dataset automatically on first run)
python src/run.py

# 5. Generate evaluation plots
python src/evaluate_model.py

# 6. Launch the web application
cd app && python app.py
```

The web application will be available at **http://localhost:5000**.

> **Note:** Pre-trained model artifacts are already included in the `models/` directory, so steps 4 and 5 are optional if you only want to run the web application.

---

## Usage

### Run the Full Training Pipeline

```bash
python src/run.py
```

This executes: data loading -> memory optimization -> cleaning -> preprocessing -> training of all 4 models -> best model selection -> artifact saving.

### Train Models Only

```bash
python src/train_model.py
```

Runs the data pipeline and training from a single module entry point.

### Evaluate the Saved Model

```bash
python src/evaluate_model.py
```

Loads `models/best_model.pkl`, re-creates the test split (deterministic via `RANDOM_STATE=42`), computes metrics, and generates confusion matrix, ROC curve, and SHAP plots in `reports/images/`.

### Run Hyperparameter Tuning

```bash
python src/tuning.py
```

Performs Cartesian grid search across all 4 models with 5-fold stratified CV. Results are saved to `models/param_search_results.json`.

### Run the Data Processing Pipeline Standalone

```bash
python src/data_processing.py
```

Loads raw data, cleans it, preprocesses it, and prints the resulting feature space and shapes.

### Launch the Web Application

```bash
cd app && python app.py
```

Default credentials: `admin` / `admin123` (auto-created on first launch).

### Run Tests

```bash
pytest tests/ -v
```

---

## Configuration

### ML Pipeline Configuration (`src/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_STATE` | `42` | Seed for reproducibility |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `CV_FOLDS` | `5` | Number of cross-validation folds |
| `MISSING_THRESHOLD` | `0.50` | Drop features with >50% NaN |
| `CORR_THRESHOLD` | `0.69` | Threshold for correlation-based imputation and feature pruning |
| `IQR_FACTOR` | `1.5` | Multiplier for IQR outlier capping |
| `MIN_CORR_TRAIN_SIZE` | `5` | Minimum non-null samples to fit regression imputation |
| `HEIGHT_CM_THRESHOLD` | `3.0` | Heuristic: if median height > 3.0, assume centimeters |
| `CATEGORY_RATIO_THRESHOLD` | `0.5` | Convert object columns to category if cardinality < 50% of rows |
| `SHAP_MAX_DISPLAY` | `15` | Number of features shown in SHAP plots |
| `SHAP_BG_SAMPLES` | `50` | Background sample size for KernelExplainer |

### Application Configuration (`app/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FLASK_SECRET_KEY` | env var or default | Secret key for session encryption |
| `MIN_USERNAME_LENGTH` | `3` | Minimum username length for registration |
| `MIN_PASSWORD_LENGTH` | `6` | Minimum password length for registration |
| `DEFAULT_ADMIN_USER` | `admin` | Auto-created admin username |
| `DEFAULT_ADMIN_PASSWORD` | `admin123` | Auto-created admin password |
| `NUMERIC_FEATURES` | 6 features | `Age`, `BMI`, `Body_Temperature`, `WBC_Count`, `CRP`, `Appendix_Diameter` |
| `BINARY_FEATURE_MAP` | 12 features | Maps form field names to one-hot encoded column names |
| `WBC_CRP_SMOOTHING` | `0.1` | Smoothing constant for WBC/CRP ratio (avoids division by zero) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FLASK_SECRET_KEY` | Override the default Flask secret key |

---

## Results

### Performance Summary

The **Random Forest** model was selected as the production model based on the recall-first selection criterion:

| Metric | Value |
|--------|-------|
| **Accuracy** | 93.63% |
| **Precision** | 91.92% |
| **Recall** | 97.85% |
| **F1 Score** | 94.79% |
| **ROC-AUC** | 97.56% |

### Full Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| SVM (RBF) | 85.99% | 92.77% | 82.80% | 87.50% | 95.24% |
| **Random Forest** | **93.63%** | 91.92% | **97.85%** | **94.79%** | 97.56% |
| LightGBM | 92.36% | 92.63% | 94.62% | 93.62% | 96.86% |
| CatBoost | 91.72% | 92.55% | 93.55% | 93.05% | **97.83%** |

### Why Random Forest Was Selected

- **Highest recall (97.85%):** In a pediatric appendicitis context, missed diagnoses (false negatives) are clinically dangerous. Recall is prioritized above all other metrics.
- **Strong overall balance:** F1 of 94.79% and precision of 91.92% show the model is not sacrificing precision excessively for recall.
- **SHAP compatibility:** Native `TreeExplainer` support means fast, exact SHAP values for every prediction.

---

## Web Application

The Flask web application provides a clinical-grade interface:

### Frontend Assets

- Global JS: `app/static/js/common.js` (navbar scroll state, smooth scroll, entry animations, flash auto-dismiss)
- Per-page JS: `landing.js`, `diagnosis.js`, `result.js`, `history.js`, `admin.js`
- CSS (in `app/static/css/`): `style.css` (global imports), `core.css` (base layout), `landing.css`, `form.css`, `result.css`, `pages.css`

### Pages

| Route | Description | Auth Required |
|-------|-------------|---------------|
| `/` | Landing page with 3-slide hero, performance stats, feature overview | No |
| `/diagnosis` | 3-step form wizard: demographics, symptoms, lab/exam results | Yes |
| `/predict` (POST) | Runs prediction, shows result with SHAP explanation | Yes |
| `/login` | User login | No |
| `/register` | User registration | No |
| `/profile` | Edit username and password | Yes |
| `/history` | Searchable and filterable prediction history | Yes |
| `/admin` | User management dashboard (toggle admin, delete users) | Admin |

### Features

- **3-step diagnostic form** with client-side validation, auto-calculated Age and BMI
- **SHAP waterfall visualization** rendered as an animated bar chart (no images -- pure HTML/CSS)
- **French-translated feature names** in SHAP explanations
- **Prediction persistence** in SQLite with patient name, timestamp, probability
- **User authentication** with bcrypt password hashing
- **Admin dashboard** for user management
- **Responsive design** with dark-mode glassmorphism UI

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_data_processing.py` | ~30 tests | Data loading, memory optimization (>30% reduction), BMI imputation, correlation imputation, cleaning (no NaN, row preservation, feature engineering), preprocessing (shapes, encoding, scaling, stratification) |
| `test_train_model.py` | ~13 tests | Model training (fit/predict methods, metric ranges), saved model validation (file existence, 21 features, predictions, metrics >= 0.85) |
| `test_evaluate_model.py` | ~9 tests | Model loading, metric computation, plot generation |
| `test_app.py` | ~26 tests | Feature vector builder (numeric fields, WBC/CRP ratio, binary toggles, peritonitis, defaults), Flask routes (auth flow, diagnosis, history, admin) |
| `test_run.py` | ~4 tests | Full pipeline integration, reproducibility, feature name consistency |
| `test_tuning.py` | ~16 tests | Model building, grid combinations, CV scoring, champion evaluation |

### CI Pipeline (`.github/workflows/ci.yml`)

The CI runs on push to `main`, tags matching `v*`, and pull requests to `main`. It includes 6 jobs:

1. **Imports** -- validate dependencies and module imports
2. **Unit Tests** -- run `pytest tests/ -v`
3. **Data Pipeline** -- run data processing and pipeline integration tests
4. **Flask Routes** -- run Flask app route tests
5. **ML Artifacts** -- validate saved model artifacts
6. **Release** -- create GitHub Release with `models/metrics.json` (on version tags only)

---

## Limitations & Future Work

### Current Limitations

- **Single dataset:** Trained and validated only on the Regensburg Pediatric Appendicitis dataset (782 patients). Generalization to other populations is not validated.
- **No external validation:** All metrics are from the same dataset split. No separate hold-out or multi-center validation.
- **Static threshold:** The classification threshold (0.498) was calibrated on this dataset and may not transfer to other clinical settings.
- **Grid search only:** The hyperparameter tuning uses exhaustive grid search, which scales combinatorially. Bayesian optimization could be more efficient.
- **No containerization:** No Dockerfile is currently provided.
- **Default admin credentials:** Hardcoded in `app/config.py` -- must be changed for any production deployment.
- **French-only UI:** The web application interface is entirely in French.

### Future Work

- Multi-center dataset validation and external test-set evaluation
- Bayesian hyperparameter optimization (e.g., Optuna)
- Model calibration analysis (reliability diagrams)
- Dockerized deployment
- Internationalization (i18n) for UI
- API endpoint for programmatic access
- Confidence interval estimation for predictions
- Temporal validation (prospective study integration)

---

## Prompt Engineering

This project was developed with AI coding agents (Claude, ChatGPT, GitHub Copilot). The full prompt engineering journal, including the prompts used at each phase, what the agents produced, and lessons learned, is available in [`docs/prompt_engineering.md`](docs/prompt_engineering.md).

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes with clear messages
4. Push to your branch: `git push origin feature/your-feature`
5. Open a Pull Request against `main`

### Guidelines

- Follow the existing code style and naming conventions
- Add tests for new functionality in the `tests/` directory
- Ensure all tests pass: `pytest tests/ -v`
- Update this README if your changes affect the pipeline, configuration, or usage

---

## License

This project was developed as part of **Coding Week 2026 -- Project 5**.

License: MIT

---

*Built by the PediAppend team -- Coding Week 2026*
