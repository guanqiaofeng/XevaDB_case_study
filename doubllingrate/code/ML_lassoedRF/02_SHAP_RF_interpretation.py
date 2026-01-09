#!/usr/bin/env python3
"""
22b_SHAP_RF_interpretation.py

SHAP interpretation pipeline for the Random Forest part of the
HVG600 Lassoed Forest model.

Generates:
 - shap_summary.png
 - shap_barplot.png
 - shap_dependence_<GENE>.png (optional)
 - shap_top_genes.csv
"""

import os
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="paper", font_scale=0.9)
plt.rcParams.update({"figure.dpi": 150})

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
model_dir = "../../data/ML_lassoedRF/HVG600_LassoForest"
in_file   = "../../data/analyze/RNAseq_with_doublingrate_800gene.csv"

out_dir = os.path.join(model_dir, "SHAP_RF")
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

# HVG600 genes
with open(os.path.join(model_dir, "HVG600_genes.txt")) as f:
    genes = [x.strip() for x in f.readlines()]

X = df[genes].copy()

# Load scaler + transform
scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
X_scaled = scaler.transform(X)

# Load RF model
rf = joblib.load(os.path.join(model_dir, "rf_model.joblib"))

print("✔ Loaded Random Forest and HVG600 data")

# ------------------------------------------------------------
# SHAP explainer (TreeExplainer for RF)
# ------------------------------------------------------------
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_scaled)

print("✔ SHAP values computed")

# ------------------------------------------------------------
# SHAP summary plot
# ------------------------------------------------------------
shap.summary_plot(
    shap_values,
    X_scaled,
    feature_names=genes,
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "SHAP_summary.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# SHAP bar plot
# ------------------------------------------------------------
shap.summary_plot(
    shap_values,
    X_scaled,
    feature_names=genes,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "SHAP_barplot.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# Save ranked SHAP genes
# ------------------------------------------------------------
mean_abs_shap = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs_shap)[::-1]

shap_df = pd.DataFrame({
    "Gene": np.array(genes)[order],
    "MeanAbsSHAP": mean_abs_shap[order]
})
shap_df.to_csv(os.path.join(out_dir, "shap_top_genes.csv"), index=False)

print("✔ Saved SHAP top genes")

# ------------------------------------------------------------
# OPTIONAL: Generate dependence plots for top 10 genes
# ------------------------------------------------------------
for g in shap_df.Gene.head(10):
    shap.dependence_plot(
        g,
        shap_values,
        X_scaled,
        feature_names=genes,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"SHAP_dependence_{g}.png"), dpi=200)
    plt.close()

print("\n🎯 SHAP interpretation completed.")
print(f"Outputs saved to: {out_dir}")
