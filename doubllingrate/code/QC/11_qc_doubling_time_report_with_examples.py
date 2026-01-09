import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
import os

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# === Input / Output ===
exp_file = "../../data/clean/experiment_clean.csv"
dt_file  = "../../data/analyze/doubling_time_per_mouse.csv"
out_pdf  = "../../data/QC/doubling_time_QC_report_with_examples.pdf"
os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

# === Load data ===
exp = pd.read_csv(exp_file)
dt  = pd.read_csv(dt_file)
exp = exp[exp["volume"] > 0]
dt  = dt[dt["doubling_time_days"] > 0]

print(f"Loaded {dt.shape[0]} mouse fits from {dt['modelID'].nunique()} models")

# === Basic stats ===
summary_stats = dt[["doubling_time_days", "r2", "n_points", "max_day_used"]].describe()
prop_good = (dt["r2"] >= 0.6).mean() * 100
prop_fast = (dt["doubling_time_days"] < 10).mean() * 100
prop_slow = (dt["doubling_time_days"] > 50).mean() * 100

# === Model-level summary ===
model_summary = (
    dt.groupby("modelID")
    .agg(mean_DT=("doubling_time_days", "mean"),
         sd_DT=("doubling_time_days", "std"),
         n_reps=("mouse_id", "nunique"),
         mean_r2=("r2", "mean"))
    .reset_index()
)

# === Helper: plot regression fit for one mouse ===
def plot_fit(mouse_id, ax_raw, ax_log):
    sub = exp[exp["model.id"] == mouse_id].sort_values("time")
    if sub.shape[0] < 3:
        return

    # raw curve
    ax_raw.plot(sub["time"], sub["volume"], marker="o", color="tab:blue")
    ax_raw.set_xlabel("Time (days)")
    ax_raw.set_ylabel("Volume (mm³)")
    ax_raw.set_title(f"{mouse_id}")

    # log fit
    x = sub["time"]
    y = np.log(sub["volume"])
    slope, intercept, r, p, se = linregress(x, y)
    y_pred = intercept + slope * x
    dt_days = np.log(2) / slope if slope > 0 else np.nan
    r2 = r**2
    ax_log.scatter(x, y, color="black", s=15)
    ax_log.plot(x, y_pred, color="red")
    ax_log.set_xlabel("Time (days)")
    ax_log.set_ylabel("log(Volume)")
    ax_log.set_title(f"{mouse_id}\nDT={dt_days:.1f}d, R²={r2:.2f}")

# === Build PDF ===
with PdfPages(out_pdf) as pdf:

    # Page 1: Summary
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    text = f"""
    Doubling Time QC Report
    =======================

    • Total mice: {dt.shape[0]}
    • Total models: {dt["modelID"].nunique()}

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

    # Page 2–6: Distributions
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(dt["doubling_time_days"], bins=40, kde=True, ax=ax)
    ax.set_xlabel("Doubling time (days)")
    ax.set_title("Distribution of Doubling Times (all mice)")
    pdf.savefig(fig); plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(dt["r2"], bins=30, kde=False, ax=ax)
    ax.set_xlabel("R² (fit quality)")
    ax.set_title("Distribution of R² Values")
    pdf.savefig(fig); plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=dt, x="r2", y="doubling_time_days", alpha=0.7, ax=ax)
    ax.set_xlabel("R²")
    ax.set_ylabel("Doubling time (days)")
    ax.set_title("Doubling Time vs. Fit Quality")
    pdf.savefig(fig); plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(model_summary["mean_DT"], bins=30, color="skyblue", kde=True, ax=ax)
    ax.set_xlabel("Mean DT (days)")
    ax.set_title("Model-level Mean Doubling Times")
    pdf.savefig(fig); plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(model_summary["sd_DT"].dropna(), bins=30, color="salmon", ax=ax)
    ax.set_xlabel("Within-model SD of DT")
    ax.set_title("Replicate Variability per Model")
    pdf.savefig(fig); plt.close()

    # === Example growth-fit plots ===
    example_ids = np.random.choice(dt["mouse_id"], size=min(8, len(dt)), replace=False)

    fig, axes = plt.subplots(nrows=len(example_ids), ncols=2, figsize=(10, 3*len(example_ids)))
    if len(example_ids) == 1:
        axes = np.array([axes])
    for i, mouse in enumerate(example_ids):
        plot_fit(mouse, axes[i,0], axes[i,1])
    plt.tight_layout()
    fig.suptitle("Example Growth-Fit Plots", y=1.02, fontsize=14)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

print(f"✅ QC report with example fits saved to:\n{out_pdf}")
