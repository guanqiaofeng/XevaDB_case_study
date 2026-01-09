import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# --- Setup ---
sns.set_style("whitegrid")

# Output directory
outdir = "../../data/QC/growth_curve_pages"
os.makedirs(outdir, exist_ok=True)

# Load data
df = pd.read_csv("../../data/clean/experiment_clean.csv")

# Extract replicate ID (e.g. m1, m2, ...)
df["replicate"] = df["model.id"].str.extract(r"\.(m\d+)$")

# All unique models
models = sorted(df["modelID"].unique())

# --- Layout config ---
n_rows, n_cols = 4, 3
plots_per_page = n_rows * n_cols
n_pages = math.ceil(len(models) / plots_per_page)

# --- Loop through pages ---
for page_idx in range(n_pages):
    start = page_idx * plots_per_page
    end = min((page_idx + 1) * plots_per_page, len(models))
    page_models = models[start:end]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 16))
    axes = axes.flatten()

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(page_models):
            ax.axis("off")
            continue

        model = page_models[ax_idx]
        sub = df[df["modelID"] == model].sort_values(["replicate", "time"])

        # Get color palette per model
        reps = sub["replicate"].unique()
        palette = sns.color_palette("tab10", n_colors=len(reps))

        for i, (rep, grp) in enumerate(sub.groupby("replicate")):
            ax.plot(grp["time"], grp["volume"], marker="o",
                    label=rep, color=palette[i])

        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Volume (mm³)")
        ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    # Add a shared legend outside the figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Replicate", loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=6, fontsize=8)

    outfile = os.path.join(outdir, f"growth_curves_page_{page_idx+1}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")
