import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os

# === Settings ===
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

infile = "../../data/analyze/doubling_time_per_mouse.csv"
out_pdf = "../../data/QC/doubling_time_QC_report.pdf"
os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

# === Load data ===
df = pd.read_csv(infile)
df = df[df["doubling_time_days"] > 0]

print(f"Loaded {df.shape[0]} mouse-level growth curves from {df['modelID'].nunique()} models")

# === Descriptive stats ===
summary_stats = df[["doubling_time_days", "r2", "n_points", "max_day_used"]].describe()

# === Model-level summary ===
model_summary = (
    df.groupby("modelID")
    .agg(
        mean_DT=("doubling_time_days", "mean"),
        sd_DT=("doubling_time_days", "std"),
        n_reps=("mouse_id", "nunique"),
        mean_r2=("r2", "mean"),
    )
    .reset_index()
)

# === QC metrics ===
prop_good = (df["r2"] >= 0.6).mean() * 100
prop_fast = (df["doubling_time_days"] < 10).mean() * 100
prop_slow = (df["doubling_time_days"] > 50).mean() * 100

# === Create PDF report ===
with PdfPages(out_pdf) as pdf:

    # Page 1: Summary text
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    text = f"""
    Doubling Time QC Report
    =======================

    File: {infile}

    • Total mice: {df.shape[0]}
    • Total models: {df["modelID"].nunique()}

    Summary statistics (doubling_time_days):
    ----------------------------------------
    {summary_stats.to_string()}

    QC Metrics:
    ------------
    • Good exponential fits (R² ≥ 0.6): {prop_good:.1f}%
    • Fast-growing tumors (DT < 10 days): {prop_fast:.1f}%
    • Slow-growing tumors (DT > 50 days): {prop_slow:.1f}%

    """
    ax.text(0, 1, text, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close()

    # Page 2: Histogram of doubling times
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["doubling_time_days"], bins=40, kde=True, ax=ax)
    ax.set_xlabel("Doubling time (days)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Doubling Times (all mice)")
    pdf.savefig(fig)
    plt.close()

    # Page 3: R² distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["r2"], bins=30, kde=False, ax=ax)
    ax.set_xlabel("R² (fit quality)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of R² Values for Exponential Fits")
    pdf.savefig(fig)
    plt.close()

    # Page 4: Scatterplot DT vs R²
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="r2", y="doubling_time_days", alpha=0.7, ax=ax)
    ax.set_xlabel("R² (fit quality)")
    ax.set_ylabel("Doubling time (days)")
    ax.set_title("Relationship between Fit Quality and Doubling Time")
    pdf.savefig(fig)
    plt.close()

    # Page 5: Per-model mean ± SD
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(model_summary["mean_DT"], bins=30, color="skyblue", kde=True, ax=ax)
    ax.set_xlabel("Mean Doubling Time (days)")
    ax.set_ylabel("Number of Models")
    ax.set_title("Distribution of Model-level Mean Doubling Times")
    pdf.savefig(fig)
    plt.close()

    # Page 6: Variation across models (SD)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(model_summary["sd_DT"].dropna(), bins=30, color="salmon", ax=ax)
    ax.set_xlabel("Within-model SD of Doubling Time")
    ax.set_ylabel("Count")
    ax.set_title("Variation of Replicate Doubling Times per Model")
    pdf.savefig(fig)
    plt.close()

print(f"✅ QC report saved to: {out_pdf}")
