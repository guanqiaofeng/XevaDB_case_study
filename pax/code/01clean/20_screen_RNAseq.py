#!/usr/bin/env python3
"""
20_screen_RNAseq.py

Clean and align RNA-seq TPM data with QC-passed PDX models.
Removes models excluded during doubling-time filtering
and produces RNA_tpm_final.csv for ML analysis.
"""

import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# --- Input / output paths
# ------------------------------------------------------------
rna_in = "../../data/procdata/RNAseq_tpm_protein_coding_hugo.csv"
qc_models = "../../data/rawdata/PDX_BR_BAR_grouped_PAX.xlsx"
out_dir = "../../data/procdata"
os.makedirs(out_dir, exist_ok=True)
rna_out = os.path.join(out_dir, "RNAseq_log2tpm_protein_coding_hugo_final.csv")

# ------------------------------------------------------------
# --- Load data
# ------------------------------------------------------------
rna = pd.read_csv(rna_in, index_col=0)
# Transpose so that model IDs are in the index (rows) and genes are columns
rna = rna.T
qc = pd.read_excel(qc_models)

# ------------------------------------------------------------
# --- Align by modelID
# ------------------------------------------------------------
qc_ids = qc["PDX_ID"].dropna().astype(str).unique().tolist()
common = rna.index.intersection(qc_ids)
rna_aligned = rna.loc[common].copy()

print(f"Total RNA models: {rna.shape[0]}")
print(f"QC-passed models: {len(qc_ids)}")
print(f"Models retained after alignment: {len(common)}")


# ------------------------------------------------------------
# --- Log2(TPM+1) transform
# ------------------------------------------------------------
rna_log = np.log2(rna_aligned + 1)

# ------------------------------------------------------------
# --- Save final matrix
# ------------------------------------------------------------
# Reset index to make modelID a proper column
rna_log_final = rna_log.reset_index()
rna_log_final.rename(columns={'index': 'modelID'}, inplace=True)
rna_log_final.to_csv(rna_out, index=False, float_format='%.2f')
print(f"✅ Saved cleaned, aligned RNA-seq matrix to:\n{rna_out}")
