#!/usr/bin/env python3
"""
21_RNAseq_QC.py

Generate QC report for final RNA-seq TPM matrix (log2-transformed).
Plots include distribution, per-sample variance, and mean–variance trend.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# ------------------------------------------------------------
# --- Setup ---
# ------------------------------------------------------------
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
    "pdf.fonttype": 42
})

rna_in = "../../data/analyze/RNAseq_log2tpm_protein_coding_hugo_final.csv"
out_dir = "../../data/analyze"
os.makedirs(out_dir, exist_ok=True)
out_pdf = os.path.join(out_dir, "RNAseq_QC_report.pdf")

# ------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------
rna = pd.read_csv(rna_in, dtype={"modelID": str})
print(f"Matrix shape: {rna.shape}")

# Separate model IDs
model_ids = rna["modelID"]
expr = rna.drop(columns="modelID")

# ------------------------------------------------------------
# --- Basic summary ---
# ------------------------------------------------------------
summary = pd.Series({
    "Total models": expr.shape[0],
    "Total genes": expr.shape[1],
    "Expression min": expr.min().min(),
    "Expression max": expr.max().max(),
    "Median expression": expr.median().median(),
    "Mean expression": expr.mean().mean(),
})

print(summary)

# ------------------------------------------------------------
# --- PDF QC Report ---
# ------------------------------------------------------------
with PdfPages(out_pdf) as pdf:

    # --- Page 1: Overview stats ---
    fig, ax = plt.subplots(figsize=(7.5, 9))
    ax.axis("off")
    text = f"""
    RNA-seq QC Report
    =================

    Input file: {rna_in}

    Summary statistics
    ------------------
    • Total models: {summary['Total models']}
    • Total genes: {summary['Total genes']}
    • Expression range: {summary['Expression min']:.2f} – {summary['Expression max']:.2f}
    • Mean expression: {summary['Mean expression']:.2f}
    • Median expression: {summary['Median expression']:.2f}

    QC objectives
    -------------
    1. Confirm expression value distribution (expected 0–12)
    2. Check sample-wise variance (no outliers)
    3. Verify mean–variance trend across genes
    4. Detect low-quality or atypical models
    """
    ax.text(0, 1, text, va="top", family="monospace", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 2: Global expression distribution ---
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(expr.values.flatten(), bins=50, color="skyblue", kde=True, ax=ax)
    ax.set_xlabel("log2(TPM + 1)")
    ax.set_ylabel("Frequency")
    ax.set_title("Global distribution of log2(TPM + 1) values")
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 3: Per-sample mean expression ---
    fig, ax = plt.subplots(figsize=(6,4))
    sample_means = expr.mean(axis=1)
    sns.boxplot(y=sample_means, color="lightgray", ax=ax)
    ax.set_ylabel("Mean log2(TPM + 1)")
    ax.set_title("Per-sample mean expression distribution")
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


    # --- Page 4: Per-sample variance ---
    fig, ax = plt.subplots(figsize=(6,4))
    sample_var = expr.var(axis=1)
    sns.boxplot(y=sample_var, color="lightgray", ax=ax)
    ax.set_ylabel("Variance across genes")
    ax.set_title("Per-sample variance (model-level)")
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 5: Mean–variance trend (gene-wise) ---
    fig, ax = plt.subplots(figsize=(6,4))
    gene_mean = expr.mean(axis=0)
    gene_var = expr.var(axis=0)
    sns.scatterplot(x=gene_mean, y=gene_var, s=10, alpha=0.3, ax=ax)
    ax.set_xlabel("Mean expression (log2 TPM)")
    ax.set_ylabel("Variance across models")
    ax.set_title("Mean–variance relationship across genes")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 6: Correlation heatmap (optional quick view) ---
    fig, ax = plt.subplots(figsize=(6,5))
    subset = expr.sample(n=min(20, expr.shape[1]), axis=1, random_state=42)
    corr = subset.corr()
    sns.heatmap(corr, cmap="vlag", ax=ax, cbar_kws={"shrink": 0.5})
    ax.set_title("Sample correlation heatmap (subset of genes)")
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

print(f"📊 RNA-seq QC report saved to:\n{out_pdf}")
