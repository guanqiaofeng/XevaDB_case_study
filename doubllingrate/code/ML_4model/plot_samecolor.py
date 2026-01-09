import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your unified HVG summary file
df = pd.read_csv("../../data/ML_4model/15_multiHVG_tuned600_800/multiHVG_summary_tuned.csv")

# Ensure consistent order of models
model_order = ["ElasticNet", "LightGBM", "RandomForest", "LassoedForest"]

# Define a shared consistent color palette
palette = {
    "ElasticNet": "#1f77b4",     # blue
    "LightGBM":   "#ff7f0e",     # orange
    "RandomForest": "#2ca02c",   # green
    "LassoedForest": "#d62728"   # red
}

# =====================================================================
# ======================     TEST R² PLOT     =========================
# =====================================================================

df_r2 = df[["Model", "n_genes", "Test_R2"]]
df_r2 = df_r2.sort_values(["Model", "n_genes"])

plt.figure(figsize=(8, 5))

sns.lineplot(
    data=df_r2,
    x="n_genes",
    y="Test_R2",
    hue="Model",
    style="Model",
    markers=True,
    dashes=False,
    linewidth=2.5,
    markeredgecolor="black",
    palette=palette,
    hue_order=model_order,
)

plt.title("Model Performance Across HVG Settings (Test R²)")
plt.xlabel("Number of Top Variable Genes (HVGs)")
plt.ylabel("Test R²")
plt.ylim(0, df_r2["Test_R2"].max() * 1.15)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("model_performance_HVG_TestR2.png", dpi=150)
plt.show()

# =====================================================================
# ====================     SPEARMAN ρ PLOT     ========================
# =====================================================================

df_sp = df[["Model", "n_genes", "Test_Spearman"]]
df_sp = df_sp.sort_values(["Model", "n_genes"])

plt.figure(figsize=(8, 5))

sns.lineplot(
    data=df_sp,
    x="n_genes",
    y="Test_Spearman",
    hue="Model",
    style="Model",
    markers=True,
    dashes=False,
    linewidth=2.5,
    markeredgecolor="black",
    palette=palette,
    hue_order=model_order,
)

plt.title("Model Rank-Order Performance Across HVG Settings (Spearman)")
plt.xlabel("Number of Top Variable Genes (HVGs)")
plt.ylabel("Test Spearman")
plt.ylim(0, df_sp["Test_Spearman"].max() * 1.15)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("model_performance_HVG_Spearman.png", dpi=150)
plt.show()
