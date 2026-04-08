"""
Standalone pipeline – runs all modules on synthetic_mobile_sales_2025.csv
and prints every result needed for the paper.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Add project root to path ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from modules.calibration_module import CalibrationAnalyzer
from modules.model_module import ModelTrainer
from modules.post_stress_module import PostStressAnalyzer
from modules.reliability_module import ReliabilityScorer
from modules.stress_module import StressTester

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  MODULE 1 – DATA PREPARATION")
print("═" * 70)

df = pd.read_csv("synthetic_mobile_sales_2025.csv")
print(f"  Dataset shape   : {df.shape}")
print(f"  Columns         : {df.columns.tolist()}")
print(f"  Missing values  : {df.isnull().sum().sum()}")
print(f"  Duplicates      : {df.duplicated().sum()}")

# ── Target: high-revenue flag (binary classification) ────────────────────
median_rev = df["Revenue_USD"].median()
df["High_Revenue"] = (df["Revenue_USD"] >= median_rev).astype(int)
print(f"\n  Target column   : High_Revenue  (1 = Revenue >= ${median_rev:,.0f})")
print(f"  Class balance   : {df['High_Revenue'].value_counts().to_dict()}")

# ── Feature engineering ───────────────────────────────────────────────────
drop_cols = ["Sale_ID", "Model", "Revenue_USD", "High_Revenue"]
cat_cols = ["Brand", "Country", "Storage", "Color", "Payment_Method"]
num_cols = ["Price_USD", "Units_Sold", "Customer_Rating", "Sale_Month", "Sale_Year"]

le = LabelEncoder()
df_enc = df.copy()
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

feature_cols = cat_cols + num_cols
X = df_enc[feature_cols].values
y = df_enc["High_Revenue"].values

# ── Train / Val / Test split (60 / 20 / 20) ──────────────────────────────
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te = scaler.transform(X_te)

print(
    f"\n  Train size : {X_tr.shape[0]}  |  Val size : {X_val.shape[0]}  |  Test size : {X_te.shape[0]}"
)

# ═══════════════════════════════════════════════════════════════════════════
# 2. BASELINE MODELING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  MODULE 2 – BASELINE MODEL TRAINING & EVALUATION")
print("═" * 70)

trainer = ModelTrainer()
model_names = ["Logistic Regression", "Random Forest", "XGBoost"]

for name in model_names:
    trainer.train_model(name, X_tr, y_tr)
    m = trainer.evaluate_model(name, X_te, y_te, dataset_name="Test")
    print(f"\n  [{name}]")
    print(f"    Accuracy  : {m['accuracy']:.4f}")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}")
    print(f"    F1-Score  : {m['f1']:.4f}")

baseline_metrics = {n: trainer.metrics[n]["Test"] for n in model_names}

# ═══════════════════════════════════════════════════════════════════════════
# 3. PREDICTION & CONFIDENCE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  MODULE 3 – PREDICTION & CONFIDENCE EXTRACTION")
print("═" * 70)

proba_dict = {}
for name in model_names:
    _, proba = trainer.predict(name, X_te)
    proba_dict[name] = proba
    conf = np.max(proba, axis=1)
    print(f"\n  [{name}]")
    print(f"    Mean confidence : {conf.mean():.4f}")
    print(f"    Std confidence  : {conf.std():.4f}")
    print(f"    Min / Max       : {conf.min():.4f} / {conf.max():.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. STRESS TESTING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  MODULE 4 – STRESS TESTING")
print("═" * 70)

stress_tester = StressTester()
stress_levels = [0.1, 0.3, 0.5]

stress_results = {n: {} for n in model_names}


# Helper ──────────────────────────────────────────────────────────────────
def eval_stressed(model_name, X_stressed, y_true, test_label):
    preds, proba = trainer.predict(model_name, X_stressed)
    acc_str = accuracy_score(y_true, preds)
    acc_ori = baseline_metrics[model_name]["accuracy"]
    agreement = np.mean(preds == baseline_metrics[model_name]["predictions"])
    result = {
        "accuracy_original": acc_ori,
        "accuracy_stressed": acc_str,
        "performance_drop": acc_ori - acc_str,
        "prediction_agreement": agreement,
        "probabilities": proba,
    }
    stress_results[model_name][test_label] = result
    return result


# ── 4a. Gaussian Noise ───────────────────────────────────────────────────
print("\n  [A] Gaussian Noise Injection")
for lvl in stress_levels:
    X_noisy = stress_tester.add_gaussian_noise(
        pd.DataFrame(X_te, columns=feature_cols), noise_level=lvl
    ).values
    for name in model_names:
        r = eval_stressed(name, X_noisy, y_te, f"gaussian_noise_{lvl}")
    print(
        f"    Noise σ={lvl} →  LR drop={stress_results['Logistic Regression'][f'gaussian_noise_{lvl}']['performance_drop']:.4f}  "
        f"RF drop={stress_results['Random Forest'][f'gaussian_noise_{lvl}']['performance_drop']:.4f}  "
        f"XGB drop={stress_results['XGBoost'][f'gaussian_noise_{lvl}']['performance_drop']:.4f}"
    )

# ── 4b. Feature Dropout ──────────────────────────────────────────────────
print("\n  [B] Feature Dropout")
for rate in [0.1, 0.2, 0.3]:
    X_drop = stress_tester.feature_dropout(
        pd.DataFrame(X_te, columns=feature_cols), dropout_rate=rate
    ).values
    for name in model_names:
        r = eval_stressed(name, X_drop, y_te, f"dropout_{rate}")
    print(
        f"    Dropout={rate} →  LR drop={stress_results['Logistic Regression'][f'dropout_{rate}']['performance_drop']:.4f}  "
        f"RF drop={stress_results['Random Forest'][f'dropout_{rate}']['performance_drop']:.4f}  "
        f"XGB drop={stress_results['XGBoost'][f'dropout_{rate}']['performance_drop']:.4f}"
    )

# ── 4c. Feature Corruption ───────────────────────────────────────────────
print("\n  [C] Feature Corruption")
for ctype in ["zero", "random", "extreme"]:
    X_corr = stress_tester.feature_corruption(
        pd.DataFrame(X_te, columns=feature_cols),
        corruption_rate=0.15,
        corruption_type=ctype,
    ).values
    for name in model_names:
        r = eval_stressed(name, X_corr, y_te, f"corruption_{ctype}")
    print(
        f"    Type={ctype} →  LR drop={stress_results['Logistic Regression'][f'corruption_{ctype}']['performance_drop']:.4f}  "
        f"RF drop={stress_results['Random Forest'][f'corruption_{ctype}']['performance_drop']:.4f}  "
        f"XGB drop={stress_results['XGBoost'][f'corruption_{ctype}']['performance_drop']:.4f}"
    )

# ── 4d. Scale Perturbation (Distribution Shift) ──────────────────────────
print("\n  [D] Scale Perturbation (Distribution Shift)")
for sf in [1.5, 2.0]:
    X_sc = stress_tester.scale_perturbation(
        pd.DataFrame(X_te, columns=feature_cols), scale_factor=sf
    ).values
    for name in model_names:
        r = eval_stressed(name, X_sc, y_te, f"scale_{sf}")
    print(
        f"    Scale={sf} →  LR drop={stress_results['Logistic Regression'][f'scale_{sf}']['performance_drop']:.4f}  "
        f"RF drop={stress_results['Random Forest'][f'scale_{sf}']['performance_drop']:.4f}  "
        f"XGB drop={stress_results['XGBoost'][f'scale_{sf}']['performance_drop']:.4f}"
    )

# ── 4e. OOD Simulation ───────────────────────────────────────────────────
print("\n  [E] Out-of-Distribution (OOD) Simulation")
X_ood = np.random.randn(*X_te.shape) * 3.0  # far from training dist
for name in model_names:
    r = eval_stressed(name, X_ood, y_te, "ood_simulation")
print(
    f"    OOD →  LR drop={stress_results['Logistic Regression']['ood_simulation']['performance_drop']:.4f}  "
    f"RF drop={stress_results['Random Forest']['ood_simulation']['performance_drop']:.4f}  "
    f"XGB drop={stress_results['XGBoost']['ood_simulation']['performance_drop']:.4f}"
)

# ═══════════════════════════════════════════════════════════════════════════
# 5. POST-STRESS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  MODULE 5 – POST-STRESS ROBUSTNESS ANALYSIS")
print("═" * 70)

post_stress = PostStressAnalyzer()
robustness_scores = {}

for name in model_names:
    post_stress.add_batch_results(stress_results[name], model_name=name)
    rob = post_stress.calculate_robustness_score(model_name=name)
    robustness_scores[name] = rob
    all_drops = [r["performance_drop"] for r in stress_results[name].values()]
    print(f"\n  [{name}]")
    print(f"    Robustness Score   : {rob:.2f} / 100")
    print(f"    Mean Acc Drop      : {np.mean(all_drops):.4f}")
    print(f"    Max Acc Drop       : {np.max(all_drops):.4f}")
    print(f"    Min Acc Drop       : {np.min(all_drops):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. CALIBRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  MODULE 6 – CALIBRATION ANALYSIS (CLEAN DATA)")
print("═" * 70)

calibrator = CalibrationAnalyzer()
cal_results_clean = {}

for name in model_names:
    proba = proba_dict[name]
    cal = calibrator.compute_calibration_metrics(y_te, proba, n_bins=10)
    cal_results_clean[name] = cal
    print(f"\n  [{name}]")
    print(f"    ECE (Expected Calibration Error) : {cal['ece']:.4f}")
    print(f"    MCE (Max Calibration Error)      : {cal['mce']:.4f}")
    print(f"    Brier Score                      : {cal['brier_score']:.4f}")
    print(f"    Avg Confidence                   : {cal['avg_confidence']:.4f}")
    print(f"    Avg Accuracy                     : {cal['avg_accuracy']:.4f}")
    print(f"    Overconfidence (conf - acc)       : {cal['overconfidence']:.4f}")

# Calibration under moderate stress (Gaussian noise 0.3)
print("\n  [Calibration under Gaussian Noise σ=0.3]")
X_noisy03 = stress_tester.add_gaussian_noise(
    pd.DataFrame(X_te, columns=feature_cols), noise_level=0.3
).values
cal_results_stress = {}
for name in model_names:
    _, proba_s = trainer.predict(name, X_noisy03)
    cal_s = calibrator.compute_calibration_metrics(y_te, proba_s, n_bins=10)
    cal_results_stress[name] = cal_s
    print(
        f"    [{name}]  ECE: {cal_s['ece']:.4f}  Overconf: {cal_s['overconfidence']:.4f}"
    )

# ═══════════════════════════════════════════════════════════════════════════
# 7. MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  MODULE 7 – MODEL COMPARISON SUMMARY")
print("═" * 70)

print(
    f"\n  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}  {'RobScore':>10}  {'ECE':>8}"
)
print("  " + "-" * 78)
for name in model_names:
    m = baseline_metrics[name]
    rs = robustness_scores[name]
    ec = cal_results_clean[name]["ece"]
    print(
        f"  {name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}  {rs:>10.2f}  {ec:>8.4f}"
    )

# ═══════════════════════════════════════════════════════════════════════════
# 8. RELIABILITY SCORING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  MODULE 8 – COMPOSITE RELIABILITY SCORING")
print("═" * 70)

scorer = ReliabilityScorer()

# Compute entropy for each model
import math

max_entropy = math.log2(2)  # binary classification

reliability_results = {}
for name in model_names:
    proba = proba_dict[name]
    # Entropy
    entropy_vals = -np.sum(proba * np.log2(proba + 1e-12), axis=1)
    avg_entropy = float(entropy_vals.mean())

    # High-confidence errors (conf > 0.8, wrong)
    conf_scores = np.max(proba, axis=1)
    preds = np.argmax(proba, axis=1)
    high_conf_mask = conf_scores >= 0.8
    if high_conf_mask.sum() > 0:
        hce_rate = np.mean(preds[high_conf_mask] != y_te[high_conf_mask])
    else:
        hce_rate = 0.0

    m = baseline_metrics[name]
    ece = cal_results_clean[name]["ece"]
    all_drops = [r["performance_drop"] for r in stress_results[name].values()]
    avg_drop = float(np.mean(all_drops))

    perf_sc = scorer._performance_score(m["accuracy"], m["f1"])
    cal_sc = scorer._calibration_score(ece)
    rob_sc = scorer._robustness_score(avg_drop)
    conf_sc = scorer._confidence_score(avg_entropy, hce_rate, max_entropy)
    total = perf_sc + cal_sc + rob_sc + conf_sc

    grade, _ = scorer._grade(total) if hasattr(scorer, "_grade") else ("N/A", "")

    reliability_results[name] = {
        "performance": perf_sc,
        "calibration": cal_sc,
        "robustness": rob_sc,
        "confidence": conf_sc,
        "total": total,
        "grade": grade,
        "avg_entropy": avg_entropy,
        "hce_rate": hce_rate,
    }

    print(f"\n  [{name}]")
    print(f"    Performance Score  : {perf_sc:.2f} / 25")
    print(f"    Calibration Score  : {cal_sc:.2f} / 25")
    print(f"    Robustness Score   : {rob_sc:.2f} / 25")
    print(f"    Confidence Score   : {conf_sc:.2f} / 25")
    print(f"    ─────────────────────────────")
    print(f"    TOTAL RELIABILITY  : {total:.2f} / 100  [Grade: {grade}]")
    print(f"    Avg Entropy        : {avg_entropy:.4f}")
    print(f"    HCE Rate           : {hce_rate:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 9. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  FINAL SUMMARY TABLE")
print("═" * 70)

print(
    f"\n  {'Model':<22} {'Acc':>6} {'F1':>6} {'ECE':>6} {'RobScore':>10} {'TotalRel':>10} {'Grade':>6}"
)
print("  " + "-" * 68)
for name in model_names:
    m = baseline_metrics[name]
    rs = robustness_scores[name]
    rr = reliability_results[name]
    print(
        f"  {name:<22} {m['accuracy']:>6.4f} {m['f1']:>6.4f} {cal_results_clean[name]['ece']:>6.4f} {rs:>10.2f} {rr['total']:>10.2f} {rr['grade']:>6}"
    )

print("\n" + "═" * 70)
print("  STRESS DEGRADATION SUMMARY (Mean Accuracy Drop per Test)")
print("═" * 70)
all_test_labels = list(stress_results["Logistic Regression"].keys())
print(f"\n  {'Stress Test':<30} {'LR':>9} {'RF':>9} {'XGB':>9}")
print("  " + "-" * 60)
for lbl in all_test_labels:
    lr_d = stress_results["Logistic Regression"][lbl]["performance_drop"]
    rf_d = stress_results["Random Forest"][lbl]["performance_drop"]
    xgb_d = stress_results["XGBoost"][lbl]["performance_drop"]
    print(f"  {lbl:<30} {lr_d:>9.4f} {rf_d:>9.4f} {xgb_d:>9.4f}")

print("\n  Done. ✓")
