#!/usr/bin/env python3
"""
Plot Test R2 (left) and Test Spearman (right) barplots for 4 models.
Bars touch (no spacing) and use identical color scheme.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("HVG600_model_summary.csv")   # update path if needed

# Ensure consistent ordering
order = ["ElasticNet", "LightGBM", "RandomForest", "LassoedForest"]
df = df.set_index("Model").loc[order].reset_index()

palette = sns.color_palette("Set2", n_colors=4)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# ================================
# LEFT — Test R2
# ================================
ax = axes[0]
sns.barplot(
    data=df,
    x="Model",
    y="Test_R2",
    palette=palette,
    ax=ax,
    edgecolor="black"
)
ax.set_title("Test R²")
ax.set_ylabel("R²")
ax.set_xlabel("")
ax.set_ylim(0, df["Test_R2"].max() * 1.25)

# Resize bars to touch each other
for container in ax.containers:
    plt.setp(container, width=0.95)

# Add values on bars
for i, v in enumerate(df["Test_R2"]):
    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# ================================
# RIGHT — Test Spearman
# ================================
ax = axes[1]
sns.barplot(
    data=df,
    x="Model",
    y="Test_Spearman",
    palette=palette,
    ax=ax,
    edgecolor="black"
)
ax.set_title("Test Spearman Correlation")
ax.set_ylabel("Spearman ρ")
ax.set_xlabel("")
ax.set_ylim(0, df["Test_Spearman"].max() * 1.25)

# Resize bars to touch each other
for container in ax.containers:
    plt.setp(container, width=0.95)

# Add values on bars
for i, v in enumerate(df["Test_Spearman"]):
    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# Rotate x labels slightly
for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig("HVG600_R2_Spearman_side_by_side_with_values.png", dpi=150)
plt.show()