"""
model_pipeline.py
CS 439 Final Project — Startup Acquisition Prediction Pipeline
  1. Load startup_data_final.csv 
  2. Time-based train/test split at 2010
  3. StandardScaler on continuous cols only
  4. Model 1 — Logistic Regression (L2, class_weight='balanced')
  5. Model 2 — XGBoost (GridSearchCV, roc_auc, cv=5)
  6. Evaluate both => Table 1 
  7. SHAP analysis on XGBoost => Figure 1
  8. PCA scatter on X_test    => Figure 2
  9. Save all plots as PNG
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)

from xgboost import XGBClassifier
import shap

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})
COLORS = {"train": "#4C72B0", "test": "#DD8452", "pos": "#55A868", "neg": "#C44E52"}

# Load data and re-attach funding year
print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

df = pd.read_csv("startup_data_final.csv")

df_raw = pd.read_csv("startup data.csv")
df_raw["first_funding_at"] = pd.to_datetime(df_raw["first_funding_at"], errors="coerce")
df["funding_year"] = df_raw["first_funding_at"].dt.year.values

print(f"  Dataset shape : {df.shape}")
print(f"  Target balance: acquired={df['target'].sum()} "
      f"({100*df['target'].mean():.1f}%)  "
      f"closed={(df['target']==0).sum()} "
      f"({100*(1-df['target'].mean()):.1f}%)")
print(f"  Funding years : {int(df['funding_year'].min())} – "
      f"{int(df['funding_year'].max())}")

# Time-based train / test split at 2010
print("\n" + "=" * 60)
print("STEP 2 — Time-based train/test split (cutoff = 2010)")
print("=" * 60)

feature_cols = [c for c in df.columns if c not in ("target", "funding_year")]
X = df[feature_cols]
y = df["target"]

train_mask = df["funding_year"] < 2010
test_mask  = df["funding_year"] >= 2010

X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
X_test,  y_test  = X[test_mask].copy(),  y[test_mask].copy()

print(f"  Train (< 2010): {len(X_train)} rows  "
      f"({100*len(X_train)/len(df):.1f}%)  "
      f"| positive rate: {y_train.mean():.2%}")
print(f"  Test  (≥ 2010): {len(X_test)} rows  "
      f"({100*len(X_test)/len(df):.1f}%)  "
      f"| positive rate: {y_test.mean():.2%}")

# Selective StandardScaler
print("\n" + "=" * 60)
print("STEP 3 — Scaling continuous features")
print("=" * 60)

CONTINUOUS_COLS = [
    "age_first_funding_year", "age_last_funding_year",
    "age_first_milestone_year", "age_last_milestone_year",
    "funding_total_usd", "relationships", "avg_participants",
    "milestones", "funding_rounds",
    "fed_rate", "vix", "sector_trend",
]
CONTINUOUS_COLS = [c for c in CONTINUOUS_COLS if c in X_train.columns]
print(f"  Scaling {len(CONTINUOUS_COLS)} continuous cols: {CONTINUOUS_COLS}")

scaler = StandardScaler()
X_train[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
X_test[CONTINUOUS_COLS]  = scaler.transform(X_test[CONTINUOUS_COLS])  

print("  ✓ Scaler fitted on train, transform applied to both splits")

# Model 1: Logistic Regression 
print("\n" + "=" * 60)
print("STEP 4 — Logistic Regression (baseline)")
print("=" * 60)

lr = LogisticRegression(
    penalty="l2",
    class_weight="balanced",   
    max_iter=1000,
    random_state=42,
    solver="lbfgs",
)
lr.fit(X_train, y_train)
print("  Logistic Regression trained")

# Model 2: XGBoost with GridSearchCV
print("\n" + "=" * 60)
print("STEP 5 — XGBoost (GridSearchCV, cv=5, scoring=roc_auc)")
print("=" * 60)
SPW = round(326 / 597, 3)
print(f"  scale_pos_weight = {SPW}  (closes/acquired = 326/597)")

param_grid = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 4, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample":     [0.8, 1.0],
}

base_xgb = XGBClassifier(
    scale_pos_weight=SPW,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

gs = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=0,
    refit=True,
)
gs.fit(X_train, y_train)

best_xgb = gs.best_estimator_
print(f"  Best params  : {gs.best_params_}")
print(f"  Best CV AUC  : {gs.best_score_:.4f}")

# Evaluate both models on X_test
print("\n" + "=" * 60)
print("STEP 6 — Evaluation on X_test  →  Table 1")
print("=" * 60)

def evaluate(model, X, y, name):
    proba = model.predict_proba(X)[:, 1]
    pred  = model.predict(X)
    return {
        "Model":    name,
        "ROC-AUC":  round(roc_auc_score(y, proba), 4),
        "F1":       round(f1_score(y, pred), 4),
        "Accuracy": round(accuracy_score(y, pred), 4),
    }

results = pd.DataFrame([
    evaluate(lr,       X_test, y_test, "Logistic Regression"),
    evaluate(best_xgb, X_test, y_test, "XGBoost"),
])
results = results.set_index("Model")

print("\n Table 1: Model Comparison ")
print(results.to_string())

for model, name in [(lr, "Logistic Regression"), (best_xgb, "XGBoost")]:
    print(f"\n  {name} — Classification Report:")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=["Closed", "Acquired"]))

# Confusion Matrix Plot 
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold", y=1.01)

for ax, (model, name) in zip(axes, [(lr, "Logistic Regression"), (best_xgb, "XGBoost")]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Closed", "Acquired"],
                yticklabels=["Closed", "Acquired"],
                cbar=False, linewidths=0.5)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("confusion_matrices.png", bbox_inches="tight")
plt.close()
print("\n  Saved => confusion_matrices.png")

# Metrics Bar Chart
fig, ax = plt.subplots(figsize=(8, 4))
results.T.plot(kind="bar", ax=ax, color=[COLORS["train"], COLORS["test"]],
               edgecolor="white", linewidth=0.8, rot=0, width=0.6)
ax.set_title("Model Comparison — Test-Set Metrics", fontsize=13, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend(title="Model", frameon=False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
for bar in ax.patches:
    ax.annotate(f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01),
                ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("model_comparison.png", bbox_inches="tight")
plt.close()
print(" Saved => model_comparison.png")

# SHAP Analysis on XGBoost =>  Figure 1
print("\n" + "=" * 60)
print("STEP 7 — SHAP Analysis (XGBoost)  →  Figure 1")
print("=" * 60)

explainer   = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

# Figure 1a: Beeswarm summary plot 
print("  Generating beeswarm summary plot …")
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_test,
    plot_type="dot",
    max_display=20,
    show=False,
)
plt.title("Figure 1 — SHAP Feature Importance (XGBoost, Test Set)",
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", bbox_inches="tight")
plt.close()
print(" Saved => shap_beeswarm.png")

# Figure 1b: Bar chart 
print("  Generating SHAP bar chart …")
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_test.columns
).sort_values(ascending=False)

top_n = 15
fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = [
    COLORS["pos"] if feat in ("fed_rate", "vix", "sector_trend") else "#6C8EBF"
    for feat in mean_abs_shap.head(top_n).index
]
bars = ax.barh(
    mean_abs_shap.head(top_n).index[::-1],
    mean_abs_shap.head(top_n).values[::-1],
    color=colors_bar[::-1],
    edgecolor="white",
    linewidth=0.6,
)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.set_title(f"Top {top_n} Features by Mean |SHAP| — XGBoost",
             fontsize=13, fontweight="bold")

macro_feats = {"fed_rate", "vix", "sector_trend"}
for bar, feat in zip(bars[::-1], mean_abs_shap.head(top_n).index):
    if feat in macro_feats:
        ax.annotate("★ macro", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=COLORS["pos"], fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["pos"], label="Macro signal (★)"),
    Patch(facecolor="#6C8EBF",    label="Other feature"),
]
ax.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("shap_bar.png", bbox_inches="tight")
plt.close()
print(" Saved => shap_bar.png")

ranking = mean_abs_shap.rank(ascending=False).astype(int)
print("\n  Macro signal rankings:")
for feat in ["fed_rate", "vix", "sector_trend"]:
    if feat in ranking:
        print(f"    {feat:15s} → rank #{ranking[feat]}  "
              f"(mean |SHAP| = {mean_abs_shap[feat]:.4f})")

# PCA Visualization on X_test  =>  Figure 2
print("\n" + "=" * 60)
print("STEP 8 — PCA (2D) scatter on X_test  →  Figure 2")
print("=" * 60)

pca = PCA(n_components=2, random_state=42)
X_test_pca = pca.fit_transform(X_test)

ev = pca.explained_variance_ratio_
print(f"  PC1 explains {ev[0]:.1%},  PC2 explains {ev[1]:.1%}  "
      f"(total {ev.sum():.1%})")

fig, ax = plt.subplots(figsize=(8, 6))
for label, color, marker, name in [
    (1, COLORS["pos"], "o", "Acquired"),
    (0, COLORS["neg"], "X", "Closed"),
]:
    mask = y_test == label
    ax.scatter(
        X_test_pca[mask, 0],
        X_test_pca[mask, 1],
        c=color, marker=marker, s=40, alpha=0.65,
        edgecolors="white", linewidths=0.4, label=f"{name} (n={mask.sum()})"
    )

ax.set_xlabel(f"PC 1 ({ev[0]:.1%} variance)", fontsize=11)
ax.set_ylabel(f"PC 2 ({ev[1]:.1%} variance)", fontsize=11)
ax.set_title("Figure 2 — PCA of Test-Set Feature Space\n"
             "(colored by acquisition outcome)",
             fontsize=13, fontweight="bold")
ax.legend(frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig("pca_scatter.png", bbox_inches="tight")
plt.close()
print(" Saved => pca_scatter.png")

# SUMMARY
print("\n" + "=" * 60)
print("PIPELINE COMPLETE — Summary")
print("=" * 60)
print("\n  Table 1 — Model Comparison:")
print(results.to_string())
print("\n  Outputs saved:")
print("    confusion_matrices.png  — Confusion matrices (both models)")
print("    model_comparison.png    — Metric bar chart (Table 1 viz)")
print("    shap_beeswarm.png       — Figure 1a: SHAP beeswarm")
print("    shap_bar.png            — Figure 1b: Mean |SHAP| bar chart")
print("    pca_scatter.png         — Figure 2: PCA scatter")
print("\n  Key finding:")
for feat in ["fed_rate", "vix", "sector_trend"]:
    if feat in ranking:
        print(f"    {feat} => SHAP rank #{ranking[feat]}")
print()
