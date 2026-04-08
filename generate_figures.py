"""
Generate all paper figures from pipeline results.
Saves PNG files to figures/ folder.
"""

import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from modules.calibration_module import CalibrationAnalyzer
from modules.model_module import ModelTrainer
from modules.post_stress_module import PostStressAnalyzer
from modules.reliability_module import ReliabilityScorer
from modules.stress_module import StressTester

np.random.seed(42)

# ── Shared style ──────────────────────────────────────────────────────────────
COLORS = {
    "Logistic Regression": "#4C78A8",
    "Random Forest": "#F58518",
    "XGBoost": "#E45756",
}
NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA + MODELS  (same pipeline as run_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("synthetic_mobile_sales_2025.csv")
median_rev = df["Revenue_USD"].median()
df["High_Revenue"] = (df["Revenue_USD"] >= median_rev).astype(int)

cat_cols = ["Brand", "Country", "Storage", "Color", "Payment_Method"]
num_cols = ["Price_USD", "Units_Sold", "Customer_Rating", "Sale_Month", "Sale_Year"]
feature_cols = cat_cols + num_cols

le = LabelEncoder()
df_enc = df.copy()
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

X = df_enc[feature_cols].values
y = df_enc["High_Revenue"].values

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

trainer = ModelTrainer()
for name in NAMES:
    trainer.train_model(name, X_tr, y_tr)
    trainer.evaluate_model(name, X_te, y_te, dataset_name="Test")

baseline_metrics = {n: trainer.metrics[n]["Test"] for n in NAMES}
proba_dict = {}
for name in NAMES:
    _, p = trainer.predict(name, X_te)
    proba_dict[name] = p

stress_tester = StressTester()
stress_results = {n: {} for n in NAMES}


def eval_stressed(model_name, X_stressed, y_true, test_label):
    preds, proba = trainer.predict(model_name, X_stressed)
    from sklearn.metrics import accuracy_score

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


for lvl in [0.1, 0.3, 0.5]:
    X_n = stress_tester.add_gaussian_noise(
        pd.DataFrame(X_te, columns=feature_cols), noise_level=lvl
    ).values
    for n in NAMES:
        eval_stressed(n, X_n, y_te, f"gaussian_noise_{lvl}")

for rate in [0.1, 0.2, 0.3]:
    X_d = stress_tester.feature_dropout(
        pd.DataFrame(X_te, columns=feature_cols), dropout_rate=rate
    ).values
    for n in NAMES:
        eval_stressed(n, X_d, y_te, f"dropout_{rate}")

for ctype in ["zero", "random", "extreme"]:
    X_c = stress_tester.feature_corruption(
        pd.DataFrame(X_te, columns=feature_cols),
        corruption_rate=0.15,
        corruption_type=ctype,
    ).values
    for n in NAMES:
        eval_stressed(n, X_c, y_te, f"corruption_{ctype}")

for sf in [1.5, 2.0]:
    X_s = stress_tester.scale_perturbation(
        pd.DataFrame(X_te, columns=feature_cols), scale_factor=sf
    ).values
    for n in NAMES:
        eval_stressed(n, X_s, y_te, f"scale_{sf}")

X_ood = np.random.randn(*X_te.shape) * 3.0
for n in NAMES:
    eval_stressed(n, X_ood, y_te, "ood_simulation")

calibrator = CalibrationAnalyzer()
cal_clean = {
    n: calibrator.compute_calibration_metrics(y_te, proba_dict[n], n_bins=10)
    for n in NAMES
}

X_noisy03 = stress_tester.add_gaussian_noise(
    pd.DataFrame(X_te, columns=feature_cols), noise_level=0.3
).values
cal_stress = {}
for name in NAMES:
    _, ps = trainer.predict(name, X_noisy03)
    cal_stress[name] = calibrator.compute_calibration_metrics(y_te, ps, n_bins=10)

post_stress = PostStressAnalyzer()
robustness_scores = {}
for n in NAMES:
    post_stress.add_batch_results(stress_results[n], model_name=n)
    robustness_scores[n] = post_stress.calculate_robustness_score(model_name=n)

scorer = ReliabilityScorer()
max_entropy = math.log2(2)
reliability_results = {}
for name in NAMES:
    proba = proba_dict[name]
    entropy_vals = -np.sum(proba * np.log2(proba + 1e-12), axis=1)
    avg_entropy = float(entropy_vals.mean())
    conf_scores = np.max(proba, axis=1)
    preds = np.argmax(proba, axis=1)
    high_conf = conf_scores >= 0.8
    hce_rate = (
        np.mean(preds[high_conf] != y_te[high_conf]) if high_conf.sum() > 0 else 0.0
    )
    m = baseline_metrics[name]
    ece = cal_clean[name]["ece"]
    avg_drop = float(
        np.mean([r["performance_drop"] for r in stress_results[name].values()])
    )
    perf_sc = scorer._performance_score(m["accuracy"], m["f1"])
    cal_sc = scorer._calibration_score(ece)
    rob_sc = scorer._robustness_score(avg_drop)
    conf_sc = scorer._confidence_score(avg_entropy, hce_rate, max_entropy)
    reliability_results[name] = {
        "performance": perf_sc,
        "calibration": cal_sc,
        "robustness": rob_sc,
        "confidence": conf_sc,
        "total": perf_sc + cal_sc + rob_sc + conf_sc,
    }

print("Data & models ready. Generating figures...")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Baseline Performance Grouped Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
metrics_list = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metrics_list))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
for i, name in enumerate(NAMES):
    vals = [baseline_metrics[name][m] for m in metrics_list]
    bars = ax.bar(
        x + i * width,
        vals,
        width,
        label=name,
        color=COLORS[name],
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0.80, 1.02)
ax.set_ylabel("Score")
ax.set_title("Fig. 2 — Baseline Model Performance Comparison (Clean Test Data)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/fig2_baseline_performance.png", bbox_inches="tight")
plt.close()
print("  ✓ Fig 2 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Stress Testing Degradation (Gaussian Noise levels)
# ═══════════════════════════════════════════════════════════════════════════════
noise_levels = [0.0, 0.1, 0.3, 0.5]
fig, ax = plt.subplots(figsize=(8, 5))
for name in NAMES:
    base_acc = baseline_metrics[name]["accuracy"]
    accs = [base_acc]
    for lvl in [0.1, 0.3, 0.5]:
        accs.append(stress_results[name][f"gaussian_noise_{lvl}"]["accuracy_stressed"])
    ax.plot(
        noise_levels,
        accs,
        marker="o",
        linewidth=2,
        markersize=7,
        label=name,
        color=COLORS[name],
    )
    for lvl_val, acc in zip(noise_levels, accs):
        ax.annotate(
            f"{acc:.3f}",
            (lvl_val, acc),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=COLORS[name],
        )

ax.set_xlabel("Gaussian Noise Level (σ)")
ax.set_ylabel("Accuracy")
ax.set_title("Fig. 3 — Accuracy Degradation Under Gaussian Noise Injection")
ax.set_ylim(0.70, 1.02)
ax.set_xticks(noise_levels)
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig3_noise_degradation.png", bbox_inches="tight")
plt.close()
print("  ✓ Fig 3 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Stress Heatmap (all stress tests × all models)
# ═══════════════════════════════════════════════════════════════════════════════
test_labels = list(stress_results["Logistic Regression"].keys())
pretty_labels = [
    l.replace("gaussian_noise_", "Noise σ=")
    .replace("dropout_", "Dropout ")
    .replace("corruption_", "Corrupt-")
    .replace("scale_", "Scale×")
    .replace("ood_simulation", "OOD")
    for l in test_labels
]

drop_matrix = np.array(
    [[stress_results[n][lbl]["performance_drop"] for lbl in test_labels] for n in NAMES]
)

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(drop_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.40)
ax.set_xticks(range(len(test_labels)))
ax.set_xticklabels(pretty_labels, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(NAMES)))
ax.set_yticklabels(NAMES, fontsize=10)
for i in range(len(NAMES)):
    for j in range(len(test_labels)):
        val = drop_matrix[i, j]
        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="black" if val < 0.25 else "white",
            fontweight="bold",
        )
plt.colorbar(im, ax=ax, label="Accuracy Drop")
ax.set_title("Fig. 4 — Stress Test Degradation Heatmap (Accuracy Drop per Condition)")
plt.tight_layout()
plt.savefig("figures/fig4_stress_heatmap.png", bbox_inches="tight")
plt.close()
print("  ✓ Fig 4 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Reliability Diagrams (3 subplots, clean data)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
for ax, name in zip(axes, NAMES):
    cal = cal_clean[name]
    bin_midpoints = (cal["bin_edges"][:-1] + cal["bin_edges"][1:]) / 2
    populated = cal["bin_count"] > 0

    # Perfect calibration diagonal
    ax.plot(
        [0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.5, label="Perfect calibration"
    )

    # Gap fill (over/under confidence)
    ax.bar(
        bin_midpoints[populated],
        cal["bin_acc"][populated],
        width=0.09,
        alpha=0.35,
        color=COLORS[name],
        label="Accuracy in bin",
    )
    ax.bar(
        bin_midpoints[populated],
        cal["bin_conf"][populated] - cal["bin_acc"][populated],
        bottom=cal["bin_acc"][populated],
        width=0.09,
        alpha=0.55,
        color="tomato",
        label="Confidence gap",
    )

    ax.plot(
        bin_midpoints[populated],
        cal["bin_acc"][populated],
        "o-",
        color=COLORS[name],
        linewidth=2,
        markersize=5,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mean Confidence")
    ax.set_title(f"{name}\nECE = {cal['ece']:.4f}")
    ax.set_aspect("equal")

axes[0].set_ylabel("Fraction of Positives (Accuracy)")
handles = [mpatches.Patch(color=COLORS[n], label=n) for n in NAMES]
handles += [
    plt.Line2D([0], [0], color="k", linestyle="--", label="Perfect"),
    mpatches.Patch(color="tomato", alpha=0.6, label="Conf. gap"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
fig.suptitle(
    "Fig. 5 — Reliability Diagrams (Confidence Calibration on Clean Test Data)",
    fontsize=13,
    y=1.02,
)
plt.tight_layout()
plt.savefig("figures/fig5_reliability_diagrams.png", bbox_inches="tight")
plt.close()
print("  ✓ Fig 5 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — ECE: Clean vs Stressed comparison
# ═══════════════════════════════════════════════════════════════════════════════
ece_clean = [cal_clean[n]["ece"] for n in NAMES]
ece_stress = [cal_stress[n]["ece"] for n in NAMES]
x = np.arange(len(NAMES))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(
    x - width / 2,
    ece_clean,
    width,
    label="Clean data",
    color="#4DBBEE",
    edgecolor="white",
)
b2 = ax.bar(
    x + width / 2,
    ece_stress,
    width,
    label="Noise stressed (σ=0.3)",
    color="#FF7F7F",
    edgecolor="white",
)
for bar, v in [
    (b, v)
    for bars, vals in [(b1, ece_clean), (b2, ece_stress)]
    for b, v in zip(bars, vals)
]:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{v:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
ax.axhline(
    0.05,
    color="green",
    linestyle=":",
    linewidth=1.4,
    alpha=0.7,
    label="ECE = 0.05 threshold",
)
ax.set_xticks(x)
ax.set_xticklabels(NAMES, fontsize=11)
ax.set_ylabel("Expected Calibration Error (ECE)")
ax.set_ylim(0, 0.18)
ax.set_title("Fig. 6 — ECE Comparison: Clean Data vs. Stressed (Gaussian Noise σ=0.3)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig6_ece_comparison.png", bbox_inches="tight")
plt.close()
print("  ✓ Fig 6 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Confidence Distribution Histograms (3 models)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, name in zip(axes, NAMES):
    proba = proba_dict[name]
    conf = np.max(proba, axis=1)
    preds = np.argmax(proba, axis=1)
    correct = preds == y_te
    ax.hist(
        conf[correct],
        bins=15,
        alpha=0.65,
        color="steelblue",
        label="Correct",
        density=True,
        edgecolor="white",
    )
    ax.hist(
        conf[~correct],
        bins=15,
        alpha=0.65,
        color="tomato",
        label="Incorrect",
        density=True,
        edgecolor="white",
    )
    ax.axvline(
        conf.mean(),
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"Mean={conf.mean():.3f}",
    )
    ax.set_xlabel("Confidence Score")
    ax.set_title(name)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Density")
fig.suptitle(
    "Fig. 7 — Prediction Confidence Distributions (Correct vs Incorrect Predictions)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("figures/fig7_confidence_distributions.png", bbox_inches="tight")
plt.close()
print("  ✓ Fig 7 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Composite Reliability Score Stacked Bar
# ═══════════════════════════════════════════════════════════════════════════════
components = ["performance", "calibration", "robustness", "confidence"]
comp_labels = ["Performance", "Calibration", "Robustness", "Confidence Quality"]
comp_colors = ["#2ECC71", "#3498DB", "#F39C12", "#9B59B6"]

fig, ax = plt.subplots(figsize=(8, 5))
bottoms = np.zeros(len(NAMES))
for comp, label, color in zip(components, comp_labels, comp_colors):
    vals = [reliability_results[n][comp] for n in NAMES]
    bars = ax.bar(
        NAMES,
        vals,
        bottom=bottoms,
        label=label,
        color=color,
        edgecolor="white",
        linewidth=0.7,
    )
    for bar, v, b in zip(bars, vals, bottoms):
        if v > 1.2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                b + v / 2,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
    bottoms += np.array(vals)

for i, name in enumerate(NAMES):
    total = reliability_results[name]["total"]
    ax.text(
        i,
        total + 0.8,
        f"Total: {total:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax.set_ylim(0, 110)
ax.set_ylabel("Reliability Score (out of 100)")
ax.set_title("Fig. 8 — Composite Reliability Score Breakdown by Component")
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1))
plt.tight_layout()
plt.savefig("figures/fig8_reliability_scores.png", bbox_inches="tight")
plt.close()
print("  ✓ Fig 8 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Radar Chart: Reliability Dimensions
# ═══════════════════════════════════════════════════════════════════════════════
categories = [
    "Performance\n(/25)",
    "Calibration\n(/25)",
    "Robustness\n(/25)",
    "Confidence\n(/25)",
]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for name in NAMES:
    vals = [reliability_results[name][c] for c in components]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", linewidth=2, label=name, color=COLORS[name])
    ax.fill(angles, vals, alpha=0.10, color=COLORS[name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0, 25)
ax.set_yticks([5, 10, 15, 20, 25])
ax.set_yticklabels(["5", "10", "15", "20", "25"], size=8)
ax.set_title(
    "Fig. 9 — Radar Chart: Multi-Dimensional Reliability Profile", size=13, pad=20
)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
plt.tight_layout()
plt.savefig("figures/fig9_radar_reliability.png", bbox_inches="tight")
plt.close()
print("  ✓ Fig 9 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — OOD vs Max Stress vs Baseline Accuracy (grouped)
# ═══════════════════════════════════════════════════════════════════════════════
categories_bar = [
    "Baseline",
    "Best Case\n(Scale 1.5×)",
    "Worst Non-OOD\n(Scale 2.0×)",
    "OOD\nSimulation",
]
data_dict = {}
for name in NAMES:
    data_dict[name] = [
        baseline_metrics[name]["accuracy"],
        stress_results[name]["scale_1.5"]["accuracy_stressed"],
        stress_results[name]["scale_2.0"]["accuracy_stressed"],
        stress_results[name]["ood_simulation"]["accuracy_stressed"],
    ]

x = np.arange(len(categories_bar))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
for i, name in enumerate(NAMES):
    bars = ax.bar(
        x + i * width,
        data_dict[name],
        width,
        label=name,
        color=COLORS[name],
        edgecolor="white",
    )
    for bar, v in zip(bars, data_dict[name]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=COLORS[name],
        )

ax.axhline(
    0.5,
    color="gray",
    linestyle=":",
    linewidth=1.2,
    alpha=0.7,
    label="Random baseline (0.5)",
)
ax.set_xticks(x + width)
ax.set_xticklabels(categories_bar, fontsize=10)
ax.set_ylim(0.40, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title(
    "Fig. 10 — Model Accuracy: Baseline vs Best-Case, Worst-Case and OOD Conditions"
)
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig10_ood_comparison.png", bbox_inches="tight")
plt.close()
print("  ✓ Fig 10 saved")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n  All figures saved to figures/")
import glob

for f in sorted(glob.glob("figures/*.png")):
    size_kb = os.path.getsize(f) // 1024
    print(f"    {f}  ({size_kb} KB)")
