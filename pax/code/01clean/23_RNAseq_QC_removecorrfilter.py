#!/usr/bin/env python3
"""
22_RNAseq_QC_v2.py

Enhanced RNA-seq QC and filtering for ML analysis.

Input: log2(TPM + 1) matrix
Implements recommended gene-screening criteria:
1. Presence filter: TPM ≥ 1 in ≥ 25% of models.
2. Top-N by variance (default N=800).
3. (Optional) Correlation pruning (|r| > 0.95).
Adds visual QC plots showing both cutoffs.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# ------------------------------------------------------------
# --- Parameters ---
# ------------------------------------------------------------
TPM_PRESENCE_THRESHOLD = 1.0
PRESENCE_FRACTION = 0.25
TOP_N = 800
CORR_PRUNE = False
CORR_THRESHOLD = 0.95

rna_in = "../../data/procdata/RNAseq_log2tpm_protein_coding_hugo_final.csv"
out_dir = "../../data/procdata"
os.makedirs(out_dir, exist_ok=True)
rna_out = os.path.join(out_dir, "RNAseq_log2tpm_filtered_for_ML_rmcorrfilter.csv")
out_pdf = os.path.join(out_dir, "RNAseq_QC_enhanced_report_v2_rmcorrfilter.pdf")

# ------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------
rna = pd.read_csv(rna_in, dtype={"modelID": str})
print(f"Input shape (log2 TPM+1): {rna.shape}")

model_ids = rna["modelID"]
expr = rna.drop(columns="modelID")

# ------------------------------------------------------------
# --- Step 1: Presence filter (using back-transformed TPM) ---
# ------------------------------------------------------------
tpm = (2 ** expr) - 1
present_fraction = (tpm >= TPM_PRESENCE_THRESHOLD).sum(axis=0) / tpm.shape[0]
keep_genes = present_fraction[present_fraction >= PRESENCE_FRACTION].index
expr_kept = expr[keep_genes]
print(f"Genes retained after presence filter: {len(keep_genes)} "
      f"({len(keep_genes)/expr.shape[1]*100:.1f}%)")

# ------------------------------------------------------------
# --- Step 2: Top-N by variance ---
# ------------------------------------------------------------
var_all = expr_kept.var(axis=0)
var_cutoff = var_all.sort_values(ascending=False).iloc[TOP_N - 1]
top_genes = var_all[var_all >= var_cutoff].index
expr_top = expr_kept[top_genes]
print(f"Top {TOP_N} variable genes retained (variance cutoff = {var_cutoff:.3f})")

# ------------------------------------------------------------
# --- Step 3: Correlation pruning (optional) ---
# ------------------------------------------------------------
if CORR_PRUNE:
    corr = expr_top.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
    expr_final = expr_top.drop(columns=to_drop)
    print(f"Correlation pruning removed {len(to_drop)} genes "
          f"(|r|>{CORR_THRESHOLD}); final genes = {expr_final.shape[1]}")
else:
    expr_final = expr_top

# ------------------------------------------------------------
# --- Save filtered dataset ---
# ------------------------------------------------------------
expr_final.insert(0, "modelID", model_ids)
expr_final.to_csv(rna_out, index=False, float_format="%.3f")
print(f"✅ Saved filtered dataset to:\n{rna_out}")

# ------------------------------------------------------------
# --- QC PDF report ---
# ------------------------------------------------------------
sns.set(style="whitegrid", context="paper", font_scale=0.9)

with PdfPages(out_pdf) as pdf:
    # Page 1: Overview
    fig, ax = plt.subplots(figsize=(7,8))
    ax.axis("off")
    text = f"""
    RNA-seq QC Report (Enhanced v2)
    ===============================

    Input file: {rna_in}

    Filters Applied
    ---------------
    • Presence: TPM ≥ {TPM_PRESENCE_THRESHOLD} in ≥ {int(PRESENCE_FRACTION*100)}% models
    • Top-{TOP_N} by variance (log2(TPM+1))
    • Correlation pruning: {'enabled' if CORR_PRUNE else 'disabled'} (|r| > {CORR_THRESHOLD})

    Summary
    --------
    • Original genes: {expr.shape[1]}
    • After presence filter: {expr_kept.shape[1]}
    • After variance selection: {expr_top.shape[1]}
    • After correlation pruning: {expr_final.shape[1]}
    • Models: {expr.shape[0]}
    """
    ax.text(0, 1, text, va="top", family="monospace", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # Page 2: Presence fraction distribution with cutoff
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(present_fraction, bins=50, color="gray", ax=ax)
    ax.axvline(PRESENCE_FRACTION, color="red", linestyle="--", label=f"Cutoff = {PRESENCE_FRACTION:.2f}")
    ax.set_xlabel("Fraction of models with TPM ≥ 1")
    ax.set_ylabel("Gene count")
    ax.set_title("Presence filter distribution")
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # Page 3: Variance distribution with cutoff
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(var_all, bins=50, color="skyblue", ax=ax)
    ax.axvline(var_cutoff, color="red", linestyle="--", label=f"Top {TOP_N} cutoff = {var_cutoff:.3f}")
    ax.set_xlabel("Gene variance (log2 TPM+1)")
    ax.set_ylabel("Gene count")
    ax.set_title("Variance filter distribution")
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # Page 4: Mean–variance trend (filtered)
    fig, ax = plt.subplots(figsize=(6,4))
    gene_mean = expr_final.drop(columns="modelID", errors="ignore").mean(axis=0)
    gene_var = expr_final.drop(columns="modelID", errors="ignore").var(axis=0)
    sns.scatterplot(x=gene_mean, y=gene_var, s=10, alpha=0.3, ax=ax)
    ax.set_xlabel("Mean log2(TPM+1)")
    ax.set_ylabel("Variance across models")
    ax.set_title("Mean–variance trend (filtered genes)")
    plt.tight_layout(pad=1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

print(f"📊 Enhanced RNA-seq QC report with cutoff plots saved to:\n{out_pdf}")
