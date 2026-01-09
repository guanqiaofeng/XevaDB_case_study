#!/usr/bin/env python3
"""
22c_Plot_Combined_Model_and_Gene_Results.py

Creates a PDF-ready multi-panel figure:

 Panels:
   A: Test R² (barplot)
   B: Test MAE (barplot)
   C: Test Spearman (barplot)
   D: SHAP top-15 RF genes (barplot)
   E: Lasso coefficient magnitudes (top 20)

Output:
   Final_ML_summary_figure.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

sns.set(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 200})

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
model_dir = "../../data/ML_lassoedRF/HVG600_LassoForest"
comp_dir = "../../data/ML_4model/HVG600_reproduce_v2"

summary_file = os.path.join(comp_dir, "HVG600_model_summary.csv")
shap_file    = os.path.join(model_dir, "SHAP_RF/shap_top_genes.csv")
lasso_coef   = os.path.join(model_dir, "LassoForest_lasso_coefficients.csv")

out_pdf = os.path.join(model_dir, "Final_ML_summary_figure.pdf")

# ------------------------------------------------------------
# Load model summary
# ------------------------------------------------------------
df = pd.read_csv(summary_file)

# Order models
model_order = ["ElasticNet", "LightGBM", "RandomForest", "LassoedForest"]
df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
df = df.sort_values("Model")

# ------------------------------------------------------------
# Load SHAP
# ------------------------------------------------------------
shap_df = pd.read_csv(shap_file).head(15)

# ------------------------------------------------------------
# Load Lasso coefficients (leaf features)
# ------------------------------------------------------------
coef_df = pd.read_csv(lasso_coef, header=None, names=["Leaf","Coef"])
coef_df["AbsCoef"] = coef_df["Coef"].abs()
coef_df = coef_df.sort_values("AbsCoef", ascending=False).head(20)

# ------------------------------------------------------------
# Multi-panel figure
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
(ax_r2, ax_mae, ax_spear,
 ax_shap, ax_lasso, ax_empty) = axes.flatten()

# ------------------------------------------------------------
# Panel A — Test R2
sns.barplot(data=df, x="Model", y="Test_R2", ax=ax_r2, palette="viridis")
ax_r2.set_title("Test R²")
ax_r2.set_ylim(0, df["Test_R2"].max()*1.2)
for i, v in enumerate(df["Test_R2"]):
    ax_r2.text(i, v+0.01, f"{v:.2f}", ha='center', va='bottom')

# ------------------------------------------------------------
# Panel B — Test MAE
sns.barplot(data=df, x="Model", y="Test_MAE", ax=ax_mae, palette="viridis")
ax_mae.set_title("Test MAE (lower = better)")
for i, v in enumerate(df["Test_MAE"]):
    ax_mae.text(i, v+0.005, f"{v:.2f}", ha='center', va='bottom')

# ------------------------------------------------------------
# Panel C — Test Spearman
sns.barplot(data=df, x="Model", y="Test_Spearman", ax=ax_spear, palette="viridis")
ax_spear.set_title("Test Spearman Rank Correlation")
ax_spear.set_ylim(0, df["Test_Spearman"].max()*1.2)
for i, v in enumerate(df["Test_Spearman"]):
    ax_spear.text(i, v+0.01, f"{v:.2f}", ha='center', va='bottom')

# ------------------------------------------------------------
# Panel D — Top SHAP Genes (RF)
sns.barplot(
    data=shap_df,
    y="Gene", x="MeanAbsSHAP",
    ax=ax_shap,
    palette="mako"
)
ax_shap.set_title("Top SHAP Genes (Random Forest)")

# ------------------------------------------------------------
# Panel E — Lasso coefficients (top 20)
sns.barplot(
    data=coef_df,
    y="Leaf", x="AbsCoef",
    ax=ax_lasso,
    palette="rocket"
)
ax_lasso.set_title("Top Lasso Leaf Coefficients")

# Remove last empty panel
ax_empty.axis("off")

plt.tight_layout()
plt.savefig(out_pdf)
plt.close()

print("\n🎯 Final multi-panel summary figure saved →")
print(out_pdf)
