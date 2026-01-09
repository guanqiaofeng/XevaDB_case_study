#!/usr/bin/env python3
"""
Three-panel bar plot:
 - Test R²
 - Test MAE
 - Test Spearman

Bars touch; values shown above bars; consistent colors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load HVG600 comparison file ===
df = pd.read_csv("HVG600_model_summary.csv")   # update path

# Ensure consistent model order
order = ["ElasticNet", "LightGBM", "RandomForest", "LassoedForest"]
df = df.set_index("Model").loc[order].reset_index()

# Color palette
palette = sns.color_palette("Set2", n_colors=4)

# Create 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# ========= Panel 1: Test R² =========
ax = axes[0]
sns.barplot(
    data=df, x="Model", y="Test_R2",
    palette=palette, edgecolor="black", ax=ax
)
ax.set_title("Test R²")
ax.set_ylabel("R²")
ax.set_xlabel("")
ax.set_ylim(0, df["Test_R2"].max() * 1.25)
# Make bars touch
for container in ax.containers:
    plt.setp(container, width=0.95)
# Values on top
for i, v in enumerate(df["Test_R2"]):
    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# ========= Panel 2: Test MAE =========
ax = axes[1]
sns.barplot(
    data=df, x="Model", y="Test_MAE",
    palette=palette, edgecolor="black", ax=ax
)
ax.set_title("Test MAE")
ax.set_ylabel("MAE")
ax.set_xlabel("")
ax.set_ylim(0, df["Test_MAE"].max() * 1.25)
for container in ax.containers:
    plt.setp(container, width=0.95)
for i, v in enumerate(df["Test_MAE"]):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# ========= Panel 3: Test Spearman =========
ax = axes[2]
sns.barplot(
    data=df, x="Model", y="Test_Spearman",
    palette=palette, edgecolor="black", ax=ax
)
ax.set_title("Test Spearman (ρ)")
ax.set_ylabel("Spearman ρ")
ax.set_xlabel("")
ax.set_ylim(0, df["Test_Spearman"].max() * 1.25)
for container in ax.containers:
    plt.setp(container, width=0.95)
for i, v in enumerate(df["Test_Spearman"]):
    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# Rotate x labels
for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig("HVG600_three_panel_performance.png", dpi=150)
plt.show()
