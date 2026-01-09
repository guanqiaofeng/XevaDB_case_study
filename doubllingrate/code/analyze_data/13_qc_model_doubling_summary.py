import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
import os

# ============================================================
# --- Setup ---
# ============================================================
sns.set(style="whitegrid", context="paper", font_scale=0.9)
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "0.3",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,   # keep text as text in PDF
})

infile = "../../data/analyze/doubling_time_per_model_final.csv"
outdir = "../../data/analyze"
os.makedirs(outdir, exist_ok=True)
out_pdf = os.path.join(outdir, "doubling_time_per_model_summary_compact.pdf")

# ============================================================
# --- Load data ---
# ============================================================
df = pd.read_csv(infile)
print(f"Loaded {len(df)} models")

# ============================================================
# --- Summary statistics ---
# ============================================================
qc_summary = pd.Series({
    "Total models": len(df),
    "Models with ≥2 replicates": (df["n_reps"] >= 2).sum(),
    "Median DT (days)": round(df["final_DT_mean"].median(), 2),
    "Mean DT (days)": round(df["final_DT_mean"].mean(), 2),
    "Fast models (DT < 10d)": (df["final_DT_mean"] < 10).sum(),
    "Slow models (DT > 40d)": (df["final_DT_mean"] > 40).sum(),
    "Mean R²": round(df["mean_r2"].mean(), 3),
})

# ============================================================
# --- Create compact PDF report ---
# ============================================================
with PdfPages(out_pdf) as pdf:

    # --- Page 1: Text summary ---
    fig, ax = plt.subplots(figsize=(7.5, 9))
    ax.axis("off")

    summary_text = f"""
    Doubling Time per Model — Summary Report
    ========================================

    Input file: {infile}

    Dataset overview
    ----------------
    • Total models: {qc_summary['Total models']}
    • Models with ≥2 replicates: {qc_summary['Models with ≥2 replicates']}
    • Mean doubling time: {qc_summary['Mean DT (days)']} days
    • Median doubling time: {qc_summary['Median DT (days)']} days
    • Mean R² (fit quality): {qc_summary['Mean R²']}
    • Fast models (DT < 10d): {qc_summary['Fast models (DT < 10d)']}
    • Slow models (DT > 40d): {qc_summary['Slow models (DT > 40d)']}

    Notes:
    -------
    • Each model’s final doubling time was derived as the mean across
      retained replicates after QC filtering.
    • Models with only 1 replicate are retained but flagged as less reliable.
    """

    ax.text(0, 1, summary_text, va="top", family="monospace", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 2: Histogram of DTs ---
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["final_DT_mean"], bins=25, kde=True, color="steelblue", ax=ax)
    ax.axvline(df["final_DT_mean"].mean(), color="red", linestyle="--", lw=0.8, label=f"Mean = {df['final_DT_mean'].mean():.1f}")
    ax.set_xlabel("Final mean doubling time (days)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of model-level doubling times")
    ax.legend(frameon=False)
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 3: SD vs mean ---
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(
        data=df, x="final_DT_mean", y="final_DT_sd",
        size="n_reps", hue="mean_r2", palette="viridis", ax=ax, alpha=0.8, legend="brief"
    )
    ax.set_xlabel("Mean doubling time (days)")
    ax.set_ylabel("SD across replicates")
    ax.set_title("Within-model variability vs mean doubling time")
    ax.legend(title="mean R²", bbox_to_anchor=(1.05,1), loc="upper left", frameon=False)
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 4: Replicate count vs doubling time ---
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(x="n_reps", y="final_DT_mean", data=df, color="lightgray", ax=ax)
    ax.set_xlabel("Replicates per model")
    ax.set_ylabel("Mean doubling time (days)")
    ax.set_title("Effect of replicate count on doubling time")
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 5: Ranked barplot ---
    df_sorted = df.sort_values("final_DT_mean")

    # Create reversed "coolwarm" palette so low = warm, high = cool
    # (red for small DTs, blue for large DTs)
    norm = mcolors.Normalize(vmin=df_sorted["final_DT_mean"].min(), 
                            vmax=df_sorted["final_DT_mean"].max())
    cmap = sns.color_palette("coolwarm_r", as_cmap=True)
    colors = [cmap(norm(v)) for v in df_sorted["final_DT_mean"]]
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x="modelID", y="final_DT_mean", data=df_sorted, palette="coolwarm", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("Model ID")
    ax.set_ylabel("Mean DT (days)")
    ax.set_title("Model-wise doubling times (sorted)")
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

print(f"📊 Compact model-level summary PDF saved to: {out_pdf}")
