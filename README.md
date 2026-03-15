# PediAppend -- AI-Powered Pediatric Appendicitis Decision Support

**Explainable machine-learning clinical decision support for pediatric appendicitis diagnosis**

[![CI Pipeline](https://github.com/Mavens06/Coding-week-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Mavens06/Coding-week-project/actions)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-1.2.4-orange.svg)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Modeling](#modeling)
- [Installation & Usage](#installation--usage)
- [Configuration](#configuration)
- [Results](#results)
- [Web Application](#web-application)
- [Testing & CI](#testing--ci)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Prompt Engineering](#prompt-engineering)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

PediAppend assists pediatricians in diagnosing appendicitis in children. It combines a Random Forest classifier with SHAP explainability to produce interpretable risk assessments.

Given clinical observations (symptoms, lab results, ultrasound findings, demographics), the system produces:

- A probability score for appendicitis
- A risk classification (high / moderate / low)
- SHAP-based explanations showing which factors drove the prediction
- A prediction history persisted per user

---

## Architecture

```
  UCI Repository (ID 938)           Local CSV
        |                               |
        +---------- load_data() --------+
                        |
                optimize_memory()          float64->float32, int64 downcast
                        |
                   clean_data()            9 steps: drop leaky columns,
                        |                  BMI reconstruction, correlation
                        |                  imputation, feature engineering,
                        |                  outlier capping, redundancy pruning
                        |
                 preprocess_data()         one-hot encoding, stratified
                        |                  80/20 split, StandardScaler
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
       save_artifacts()  ->  Flask web app (app/app.py)
                             Per-patient prediction + SHAP
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Regensburg Pediatric Appendicitis](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis) (UCI #938) |
| Patients | 782 |
| Raw features | 53 |
| Target | `Diagnosis` (appendicitis vs. no appendicitis) |
| Class balance | ~59/41% (approximately balanced) |

The dataset is auto-downloaded from UCI on first run via `ucimlrepo` and cached at `data/appendicitis.csv`.

### Preprocessing (9 steps in `clean_data()`)

| Step | Operation |
|------|-----------|
| 1 | Drop leak-prone, circular, and weak columns |
| 2 | Drop features with >50% missing values |
| 3 | Reconstruct BMI from Height/Weight, then drop Height and Weight |
| 4 | Linear regression imputation for correlated pairs (\|r\| >= 0.69) |
| 5 | Feature engineering: `WBC_CRP_Ratio`, `Neutrophil_WBC_Interaction` |
| 6 | Median imputation (numeric) + mode imputation (categorical) |
| 7 | Deduplication |
| 8 | IQR outlier capping (1.5x IQR) |
| 9 | Greedy removal of correlated features (\|r\| >= 0.69) |

After cleaning: 21 features. Encoding: one-hot for categoricals, `StandardScaler` fit on train only.

---

## Modeling

Four models are compared. The best is selected by lexicographic ordering: recall > precision > ROC-AUC. Recall-first because missing an appendicitis diagnosis is clinically dangerous.

| Model | Key Hyperparameters |
|-------|---------------------|
| SVM (RBF) | `C=1.0`, `gamma=scale`, `class_weight=balanced` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=2` |
| LightGBM | `n_estimators=100`, `max_depth=6`, `learning_rate=0.05` |
| CatBoost | `iterations=200`, `depth=10`, `learning_rate=0.01` |

SHAP explainability: `TreeExplainer` for tree models, `KernelExplainer` (50 background samples) for SVM.

---

## Installation & Usage

### Quick start

```bash
git clone https://github.com/Mavens06/Coding-week-project.git
cd Coding-week-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pre-trained model artifacts are included in `models/`, so you can skip training and go straight to the web app.

### Commands

| Command | Description |
|---------|-------------|
| `python src/run.py` | Full pipeline: load, clean, preprocess, train, save |
| `python src/train_model.py` | Train only (also runs data pipeline) |
| `python src/evaluate_model.py` | Generate metrics, confusion matrix, ROC curve, SHAP plots |
| `python src/tuning.py` | Grid search across all 4 models (5-fold CV) |
| `cd app && python app.py` | Launch the web app on http://localhost:5000 |
| `pytest tests/ -v` | Run the test suite |

### Default admin credentials

| | |
|--|--|
| Username | `admin` |
| Password | `admin123` |

Auto-created on first launch. Change these in `app/config.py` for any non-local deployment.

---

## Configuration

### ML pipeline (`src/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `MISSING_THRESHOLD` | `0.50` | Drop features with >50% NaN |
| `CORR_THRESHOLD` | `0.69` | Correlation-based imputation and pruning |
| `IQR_FACTOR` | `1.5` | Outlier capping multiplier |

### Application (`app/config.py`)

| Parameter | Default |
|-----------|---------|
| `DEFAULT_ADMIN_USER` | `admin` |
| `DEFAULT_ADMIN_PASSWORD` | `admin123` |
| `MIN_USERNAME_LENGTH` | `3` |
| `MIN_PASSWORD_LENGTH` | `6` |
| `FLASK_SECRET_KEY` | env var `FLASK_SECRET_KEY` or built-in default |

---

## Results

### Selected model: Random Forest

| Metric | Value |
|--------|-------|
| Accuracy | 93.63% |
| Precision | 91.92% |
| Recall | **97.85%** |
| F1 Score | 94.79% |
| ROC-AUC | 97.56% |

### Full comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| SVM (RBF) | 85.99% | 92.77% | 82.80% | 87.50% | 95.24% |
| **Random Forest** | **93.63%** | 91.92% | **97.85%** | **94.79%** | 97.56% |
| LightGBM | 92.36% | 92.63% | 94.62% | 93.62% | 96.86% |
| CatBoost | 91.72% | 92.55% | 93.55% | 93.05% | **97.83%** |

Random Forest was selected for highest recall (97.85%). In medical diagnosis, false negatives are more costly than false positives.

### Saved artifacts (`models/`)

| File | Contents |
|------|----------|
| `best_model.pkl` | Random Forest model |
| `scaler.pkl` | Fitted StandardScaler |
| `feature_names.pkl` | 21 feature names |
| `metrics.json` | Metrics for all 4 models |
| `threshold.txt` | Classification threshold (0.498) |
| `param_search_results.json` | Grid search results |

---

## Web Application

### Routes

| Route | Description | Auth |
|-------|-------------|------|
| `/` | Landing page | No |
| `/diagnosis` | 3-step form wizard (demographics, symptoms, lab/exam) | Yes |
| `/predict` (POST) | Prediction + SHAP explanation | Yes |
| `/login`, `/register` | Authentication | No |
| `/profile` | Edit username/password | Yes |
| `/history` | Searchable prediction history | Yes |
| `/admin` | User management (toggle admin, delete users) | Admin |

### Features

- 3-step diagnostic form with client-side validation, auto-calculated Age and BMI
- SHAP waterfall rendered as animated HTML/CSS bar chart (no matplotlib images)
- French-translated feature names in SHAP explanations
- Prediction persistence in SQLite
- User authentication with bcrypt password hashing
- Responsive dark-mode glassmorphism UI

### Frontend files

- CSS (in `app/static/css/`): `style.css`, `core.css`, `landing.css`, `form.css`, `result.css`, `pages.css`
- JS (in `app/static/js/`): `common.js`, `landing.js`, `diagnosis.js`, `result.js`, `history.js`, `admin.js`

---

## Testing & CI

```bash
pytest tests/ -v
```

| Module | Tests | What it covers |
|--------|-------|----------------|
| `test_data_processing.py` | ~30 | Loading, memory optimization, cleaning, preprocessing |
| `test_train_model.py` | ~13 | Training, model selection, saved artifacts |
| `test_evaluate_model.py` | ~9 | Metrics, plot generation |
| `test_app.py` | ~26 | Feature vector builder, Flask routes, auth flow |
| `test_run.py` | ~4 | Pipeline integration, reproducibility |
| `test_tuning.py` | ~16 | Grid search, CV scoring |

### CI pipeline (`.github/workflows/ci.yml`)

Triggered on push to `main`, PRs to `main`, and `v*` tags. 6 jobs:

1. Imports validation
2. Unit tests
3. Data pipeline tests
4. Flask route tests
5. ML artifact validation
6. GitHub Release (on `v*` tags only, attaches `models/metrics.json`)

---

## Project Structure

```
Coding-week-project/
|
|-- app/                                 # Flask web application
|   |-- app.py                           # Routes (/, /diagnosis, /predict)
|   |-- auth.py                          # Auth blueprint (register, login, admin)
|   |-- config.py                        # App constants, feature maps, auth defaults
|   |-- shap_utils.py                    # SHAP explainer init + per-prediction values
|   |-- static/
|   |   |-- css/                         # core, form, landing, pages, result, style
|   |   |-- img/                         # anatomy images, favicon
|   |   `-- js/                          # common, landing, diagnosis, result, history, admin
|   `-- templates/                       # base, index, diagnosis, result, login, register,
|                                        # profile, history, admin
|
|-- data/
|   |-- appendicitis.csv                 # Raw dataset (782 patients, 53 features)
|   `-- processed/                       # X_processed.csv, y_processed.csv
|
|-- docs/
|   `-- prompt_engineering.md            # Prompt engineering journal
|
|-- models/                              # Trained artifacts (pkl, json, txt)
|-- notebooks/
|   `-- eda.ipynb                        # Exploratory data analysis (41 cells)
|
|-- reports/
|   |-- figures/                         # EDA visualizations (from notebook)
|   `-- images/                          # Model evaluation plots (from evaluate_model.py)
|
|-- src/                                 # ML pipeline
|   |-- config.py                        # Central configuration
|   |-- data_processing.py               # load, optimize, clean (9 steps), preprocess
|   |-- train_model.py                   # Train 4 models, CV, select best, save
|   |-- evaluate_model.py                # Metrics, confusion matrix, ROC, SHAP plots
|   |-- tuning.py                        # Grid search
|   `-- run.py                           # Pipeline entry point
|
|-- tests/                               # Pytest suite (~80 tests)
|-- requirements.txt
`-- README.md
```

---

## Limitations

- Trained on a single dataset (782 patients). No external validation.
- Static classification threshold (0.498), calibrated on this dataset.
- Grid search only (no Bayesian optimization).
- Default admin credentials hardcoded -- change for production.
- French-only UI.

---

## Prompt Engineering

This project was developed with AI coding agents (Claude, ChatGPT, GitHub Copilot). The full journal of prompts, results, and lessons learned is in [`docs/prompt_engineering.md`](docs/prompt_engineering.md).

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes
4. Push and open a Pull Request against `main`
5. Make sure `pytest tests/ -v` passes

---

## License

Developed as part of Coding Week 2026 -- Project 5. License: MIT.
