# ML Model Reliability and Stress Testing Framework

A Streamlit-based framework for evaluating classification models across performance, confidence, calibration, stress robustness, reliability scoring, and report generation.

## What Is Implemented

The app currently exposes **9 modules** in the sidebar:

1. Data Management
2. Baseline Modeling
3. Prediction and Confidence
4. Stress Testing
5. Post-Stress Evaluation
6. Calibration Analysis
7. Model Comparison
8. Reliability Scoring
9. Visualization and Reports

## Module Status (Current)

| Module | Status | Core capabilities |
|---|---|---|
| 1. Data Management | Complete | CSV upload, validation, cleaning, encoding, scaling, train/val/test split |
| 2. Baseline Modeling | Complete | Logistic Regression, Random Forest, XGBoost, evaluation, confusion matrix, feature importance, save/load |
| 3. Prediction and Confidence | Complete | Predictions, confidence analysis, high-confidence errors, calibration/entropy analysis |
| 4. Stress Testing | Complete | Single and batch perturbation tests (noise, dropout, corruption, scaling, distribution shift) |
| 5. Post-Stress Evaluation | Complete | Robustness score, vulnerability analysis, stress-type summary, recommendations |
| 6. Calibration Analysis | Complete | Reliability diagrams, calibration metrics, per-class analysis, temperature scaling |
| 7. Model Comparison | Complete | Cross-model performance tables/charts, robustness and calibration comparison, best-model recommendation |
| 8. Reliability Scoring | Complete | Unified 0-100 reliability score with component breakdown, grades, per-model recommendations |
| 9. Visualization and Reports | Complete | Unified dashboard and export to CSV/JSON/HTML/PDF |

## Quick Start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### 2) Run the app

```bash
streamlit run app.py
```

Open the URL shown by Streamlit (usually http://localhost:8501).

## Recommended Workflow

1. Use Module 1 to prepare data.
2. Train and evaluate models in Module 2.
3. Inspect confidence behavior in Module 3.
4. Run perturbation tests in Module 4.
5. Review robustness findings in Module 5.
6. Analyze and improve calibration in Module 6.
7. Compare all trained models in Module 7.
8. Use reliability scoring in Module 8.
9. Export consolidated outputs in Module 9.

## Project Structure

```text
Mini Project/
|- app.py
|- requirements.txt
|- modules/
|  |- data_module.py
|  |- model_module.py
|  |- stress_module.py
|  |- post_stress_module.py
|  |- calibration_module.py
|  |- comparison_module.py
|  |- reliability_module.py
|  |- reporting_module.py
|- utils/
|  |- metrics.py
|  |- plotting.py
`- saved_models/                # local model artifacts
```

## Artifacts and Git Tracking

Generated/testing artifacts are intentionally excluded from version control:

- `saved_models/` (trained model binaries)
- `figures/` (paper/testing output images)
- personal/local datasets (for example `synthetic_mobile_sales_2025.csv`)
- paper-specific helper scripts and document files

These are controlled via `.gitignore`.

## Key Dependencies

- streamlit
- scikit-learn
- xgboost
- pandas
- numpy
- plotly
- matplotlib
- seaborn
- shap
- reportlab

## Notes

- This project is focused on classification workflows with probability outputs.
- Some calibration/reliability views require models that support `predict_proba`.
- Robustness and reliability views are most informative after batch stress tests are run.
