#!/usr/bin/env python3
"""
40_screen_genes_for_ML.py

Screen RNAseq genes (features) before ML modeling.
Applies variance and expression filters, and generates a QC summary PDF.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# ------------------------------------------------------------
# --- Paths
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate.csv"
out_dir = "../../data/analyze"
os.makedirs(out_dir, exist_ok=True)

filtered_out = os.path.join(out_dir, "RNAseq_with_doublingrate_filtered.csv")
qc_pdf = os.path.join(out_dir, "RNAseq_gene_filter_QC.pdf")

# ------------------------------------------------------------
# --- Load data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]
X = df.drop(columns=meta_cols)
print(f"Input shape: {X.shape}")

# ------------------------------------------------------------
# --- Filtering thresholds
# ------------------------------------------------------------
var_cutoff = 0.01
mean_cutoff = 1.0

# ------------------------------------------------------------
# --- Compute stats
# ------------------------------------------------------------
gene_var = X.var(axis=0)
gene_mean = X.mean(axis=0)

# Variance and mean filters
mask = (gene_var > var_cutoff) & (gene_mean > mean_cutoff)
X_filtered = X.loc[:, mask]
filtered = pd.concat([df[meta_cols], X_filtered], axis=1)

print(f"Genes before filtering: {X.shape[1]}")
print(f"Genes retained: {X_filtered.shape[1]} ({100*X_filtered.shape[1]/X.shape[1]:.1f}%)")

# ------------------------------------------------------------
# --- Save filtered dataset
# ------------------------------------------------------------
filtered.to_csv(filtered_out, index=False, float_format="%.3f")
print(f"✅ Saved filtered dataset to:\n{filtered_out}")

# ------------------------------------------------------------
# --- Generate QC PDF
# ------------------------------------------------------------
sns.set(style="whitegrid", context="paper", font_scale=0.9)
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42
})

with PdfPages(qc_pdf) as pdf:
    # Page 1: Summary
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis("off")
    text = f"""
    RNA-seq Gene Filtering QC Report
    =================================

    Input file: {in_file}

    Filters applied:
    ----------------
    • Variance > {var_cutoff}
    • Mean expression > {mean_cutoff}

    Summary:
    --------
    • Models: {X.shape[0]}
    • Genes before filtering: {X.shape[1]}
    • Genes retained: {X_filtered.shape[1]}
      ({100*X_filtered.shape[1]/X.shape[1]:.1f}% of total)

    Notes:
    ------
    • Variance threshold removes near-constant genes.
    • Mean expression threshold removes lowly expressed genes.
    • Next step: modeling with ElasticNet / RF using final_DT_mean.
    """
    ax.text(0, 1, text, va="top", family="monospace", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # Page 2: Variance histogram
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(gene_var, bins=50, color="skyblue", ax=ax)
    ax.axvline(var_cutoff, color="red", linestyle="--", label="Variance cutoff")
    ax.set_xlabel("Gene variance (log2 TPM)")
    ax.set_ylabel("Number of genes")
    ax.set_title("Distribution of gene variance")
    ax.legend(frameon=False)
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # Page 3: Mean expression histogram
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(gene_mean, bins=50, color="lightgreen", ax=ax)
    ax.axvline(mean_cutoff, color="red", linestyle="--", label="Mean cutoff")
    ax.set_xlabel("Mean log2(TPM + 1)")
    ax.set_ylabel("Number of genes")
    ax.set_title("Distribution of gene mean expression")
    ax.legend(frameon=False)
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # Page 4: Variance vs. Mean scatter
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=gene_mean, y=gene_var, alpha=0.3, s=10, ax=ax)
    ax.axvline(mean_cutoff, color="red", linestyle="--")
    ax.axhline(var_cutoff, color="red", linestyle="--")
    ax.set_xlabel("Mean log2(TPM + 1)")
    ax.set_ylabel("Variance (log2 TPM)")
    ax.set_title("Mean–variance relationship")
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

print(f"📊 QC report saved to:\n{qc_pdf}")
