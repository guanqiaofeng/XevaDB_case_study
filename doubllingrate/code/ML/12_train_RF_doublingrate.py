#!/usr/bin/env python3
"""
12_train_RF_doublingrate.py

Train a Random Forest Regressor to predict PDX doubling time from RNA-seq data.

Implements:
 - Weighted regression using inverse SD as reliability weight
 - Feature preselection by variance
 - 5-fold cross-validation with R², MAE, Spearman metrics
 - Feature importance and SHAP interpretation

Outputs:
 - RF_model_performance_summary.csv
 - RF_feature_importance.csv
 - Observed_vs_Predicted_RF.png
 - SHAP_summary_plot.png
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import shap

# ------------------------------------------------------------
# --- Configurable parameters
# ------------------------------------------------------------
N_TOP_GENES = 1000         # Number of most variable genes to use
N_TREES = 1000             # Number of trees in the forest
RANDOM_STATE = 42
OUT_DIR = "../../data/ML"

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# --- Load data
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_filtered.csv"
df = pd.read_csv(in_file, dtype={"modelID": str})

meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]
X = df.drop(columns=meta_cols)
y = df["final_DT_mean"]
sd = df["final_DT_sd"].copy()

# ------------------------------------------------------------
# --- Sample weighting
# ------------------------------------------------------------
sd = sd.replace(0, np.nan)
weights = 1 / (sd + 1e-3)
min_weight = np.nanpercentile(weights, 10)
weights = weights.fillna(min_weight)
weights = weights / weights.mean()

print(f"Sample weights range: {weights.min():.3f} – {weights.max():.3f}")

# ------------------------------------------------------------
# --- Feature selection by variance
# ------------------------------------------------------------
gene_var = X.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).head(N_TOP_GENES).index
X_sel = X[top_genes].copy()
print(f"Selected top {N_TOP_GENES} high-variance genes for modeling.")

# Optionally scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

# ------------------------------------------------------------
# --- Train Random Forest with 5-fold CV
# ------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
r2_list, mae_list, sp_list = [], [], []

rf = RandomForestRegressor(
    n_estimators=N_TREES,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_features="sqrt",
    min_samples_leaf=2
)

y_pred_all = np.zeros_like(y)

for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled)):
    rf.fit(X_scaled[train_idx], y.iloc[train_idx], sample_weight=weights.iloc[train_idx])
    y_pred = rf.predict(X_scaled[test_idx])

    y_pred_all[test_idx] = y_pred

    r2 = r2_score(y.iloc[test_idx], y_pred)
    mae = mean_absolute_error(y.iloc[test_idx], y_pred)
    sp = spearmanr(y.iloc[test_idx], y_pred).correlation
    r2_list.append(r2)
    mae_list.append(mae)
    sp_list.append(sp)

    print(f"Fold {fold+1}: R²={r2:.3f}, MAE={mae:.2f}, Spearman={sp:.3f}")

# ------------------------------------------------------------
# --- Summary metrics
# ------------------------------------------------------------
results = pd.DataFrame({
    "Metric": ["R2", "MAE", "Spearman"],
    "Mean": [np.mean(r2_list), np.mean(mae_list), np.mean(sp_list)],
    "SD": [np.std(r2_list), np.std(mae_list), np.std(sp_list)]
})

results.to_csv(os.path.join(OUT_DIR, "RF_model_performance_summary.csv"), index=False)
print(f"✅ Saved performance summary → {OUT_DIR}/RF_model_performance_summary.csv")

# ------------------------------------------------------------
# --- Observed vs Predicted plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5,4))
sns.regplot(x=y, y=y_pred_all, scatter_kws={'s':40, 'alpha':0.7})
ax.set_xlabel("Observed Doubling Time (days)")
ax.set_ylabel("Predicted (Random Forest)")
ax.set_title(f"Observed vs Predicted (R²={np.mean(r2_list):.2f}, MAE={np.mean(mae_list):.1f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Observed_vs_Predicted_RF.png"), dpi=150)
plt.close()

# ------------------------------------------------------------
# --- Feature importance
# ------------------------------------------------------------
rf.fit(X_scaled, y, sample_weight=weights)
feat_importance = pd.Series(rf.feature_importances_, index=top_genes).sort_values(ascending=False)
feat_importance.to_csv(os.path.join(OUT_DIR, "RF_feature_importance.csv"))
print(f"✅ Saved feature importance → {OUT_DIR}/RF_feature_importance.csv")

# ------------------------------------------------------------
# --- SHAP interpretation
# ------------------------------------------------------------
print("Computing SHAP values (may take a few minutes)...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_scaled)

plt.figure(figsize=(6,4))
shap.summary_plot(shap_values, X_sel, show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "SHAP_summary_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved SHAP summary plot → {OUT_DIR}/SHAP_summary_plot.png")

print("🎯 Random Forest training and interpretation complete.")
