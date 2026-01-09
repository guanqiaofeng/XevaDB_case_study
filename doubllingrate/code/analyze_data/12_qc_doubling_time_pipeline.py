import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

# ============================================================
# --- Setup ---
# ============================================================
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

infile = "../../data/analyze/doubling_time_per_mouse.csv"
outdir_qc = "../../data/QC"
outdir_result = "../../data/analyze"
os.makedirs(outdir_qc, exist_ok=True)
os.makedirs(outdir_result, exist_ok=True)

out_mouse = os.path.join(outdir_result, "doubling_time_per_mouse_clean.csv")
out_model = os.path.join(outdir_result, "doubling_time_per_model_final.csv")
out_pdf   = os.path.join(outdir_qc, "doubling_time_QC_pipeline_report.pdf")

# ============================================================
# --- Load data ---
# ============================================================
df = pd.read_csv(infile)
print(f"Loaded {df.shape[0]} mouse-level fits across {df['modelID'].nunique()} models")

# ============================================================
# 1️⃣ Per-replicate filtering
# ============================================================
# Apply replicate-level QC
mask_replicate = (
    (df["r2"] >= 0.6) &
    (df["doubling_time_days"] >= 2.5) &
    (df["doubling_time_days"] <= 50)
)

df_filt = df[mask_replicate].copy()
removed_replicate = df.loc[~mask_replicate, "mouse_id"].tolist()

print(f"After per-replicate filter: {df_filt.shape[0]} mice retained "
      f"({100 * df_filt.shape[0] / df.shape[0]:.1f}%)")

# ============================================================
# 2️⃣ Per-model consistency filtering
# ============================================================
summary = (
    df_filt.groupby("modelID")["doubling_time_days"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "mean_DT", "std": "sd_DT", "count": "n_reps"})
)

df_merged = df_filt.merge(summary, on="modelID", how="left")

df_merged["is_outlier"] = (
    (df_merged["n_reps"] >= 2) &
    (np.abs(df_merged["doubling_time_days"] - df_merged["mean_DT"]) > 2 * df_merged["sd_DT"])
)

removed_model_outlier = df_merged.loc[df_merged["is_outlier"], "mouse_id"].tolist()
n_outliers = len(removed_model_outlier)

clean_df = df_merged[~df_merged["is_outlier"]].copy()
print(f"Flagged {n_outliers} intra-model outlier replicates")

# ============================================================
# 3️⃣ Recompute final per-model summary
# ============================================================
final_model_df = (
    clean_df.groupby("modelID")
    .agg(
        final_DT_mean=("doubling_time_days", "mean"),
        final_DT_sd=("doubling_time_days", "std"),
        n_reps=("mouse_id", "nunique"),
        mean_r2=("r2", "mean")
    )
    .reset_index()
)

# ============================================================
# 4️⃣ Save outputs
# ============================================================
clean_df.to_csv(out_mouse, index=False)
final_model_df.to_csv(out_model, index=False)
print(f"✅ Cleaned data saved to:\n  {out_mouse}\n  {out_model}")

# ============================================================
# 5️⃣ Generate PDF QC Report (with formatted ID lists)
# ============================================================

def format_id_list(id_list, per_line=4, max_total=80):
    """Return a formatted multi-line string (4 IDs per line, truncated if long)."""
    if len(id_list) == 0:
        return "None"
    shown_ids = id_list[:max_total]
    lines = []
    for i in range(0, len(shown_ids), per_line):
        lines.append(", ".join(shown_ids[i:i+per_line]))
    if len(id_list) > max_total:
        lines.append(f"... (+{len(id_list) - max_total} more)")
    return "\n".join(lines)


with PdfPages(out_pdf) as pdf:
    # --- Page 1: Summary text + removed IDs (formatted neatly) ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    text_header = f"""
    Doubling Time QC Pipeline Report
    ================================

    Input file: {infile}

    • Total mice (input): {df.shape[0]}
    • Models: {df['modelID'].nunique()}

    After per-replicate filters:
    ----------------------------
    • Retained: {df_filt.shape[0]} mice ({100 * df_filt.shape[0] / df.shape[0]:.1f}%)
    • Criteria: R² ≥ 0.6,  2.5 ≤ DT ≤ 50 days
    """

    replicate_ids_text = format_id_list(removed_replicate)
    model_outlier_text = format_id_list(removed_model_outlier)

    text_ids = f"""
    Removed mouse IDs (replicate filter):
    -------------------------------------
    {replicate_ids_text}

    After per-model consistency filter:
    -----------------------------------
    • Outliers flagged: {n_outliers}
    • Final retained mice: {clean_df.shape[0]}
    • Final models: {final_model_df.shape[0]}

    Removed mouse IDs (model-level outlier filter):
    -----------------------------------------------
    {model_outlier_text}
    """

    ax.text(0, 1, text_header + text_ids, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Distribution before vs after cleaning ---
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["doubling_time_days"], bins=40, kde=True, color="gray", label="Raw", ax=ax)
    sns.histplot(clean_df["doubling_time_days"], bins=40, kde=True, color="skyblue", label="Clean", ax=ax)
    ax.set_xlabel("Doubling Time (days)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Doubling Times (Before vs After QC)")
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: R² distribution ---
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["r2"], bins=30, color="gray", label="Raw", kde=False, ax=ax)
    sns.histplot(clean_df["r2"], bins=30, color="tab:green", label="Clean", kde=False, ax=ax)
    ax.set_xlabel("R² (Fit quality)")
    ax.set_title("Distribution of R² Before vs After QC")
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    # --- Page 4: DT vs R² scatter (colored by outlier) ---
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(
        data=df_merged, x="r2", y="doubling_time_days",
        hue="is_outlier", palette={False: "blue", True: "red"},
        alpha=0.7, ax=ax
    )
    ax.set_xlabel("R²")
    ax.set_ylabel("Doubling Time (days)")
    ax.set_title("Outlier Flagging: Red = Removed")
    pdf.savefig(fig)
    plt.close()

    # --- Page 5: Model-level SD vs Mean ---
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=summary, x="mean_DT", y="sd_DT", size="n_reps", ax=ax)
    ax.set_xlabel("Mean DT (days)")
    ax.set_ylabel("SD across replicates")
    ax.set_title("Intra-model variability")
    pdf.savefig(fig)
    plt.close()

    # --- Page 6: Histogram of final model DTs ---
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(final_model_df["final_DT_mean"], bins=30, color="tab:blue", kde=True, ax=ax)
    ax.set_xlabel("Final Model-level Doubling Time (days)")
    ax.set_title("Final Cleaned Model Distribution")
    pdf.savefig(fig)
    plt.close()

print(f"📄 PDF QC report saved to: {out_pdf}")
