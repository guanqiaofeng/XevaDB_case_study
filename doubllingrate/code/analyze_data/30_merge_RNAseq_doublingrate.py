#!/usr/bin/env python3
"""
30_merge_RNAseq_doublingrate.py

Merge RNAseq TPM matrix with doubling-time metadata
for ML modeling (step 1 & 2 only).
"""

import pandas as pd
import os

# ------------------------------------------------------------
# --- Input / output paths
# ------------------------------------------------------------
rna_in = "../../data/analyze/RNAseq_log2tpm_filtered_for_ML.csv"
dt_in  = "../../data/analyze/doubling_time_per_model_final.csv"
out_dir = "../../data/analyze"
os.makedirs(out_dir, exist_ok=True)
merged_out = os.path.join(out_dir, "RNAseq_with_doublingrate_797gene.csv")

# ------------------------------------------------------------
# --- Load data
# ------------------------------------------------------------
rna = pd.read_csv(rna_in, dtype={"modelID": str})
dt  = pd.read_csv(dt_in, dtype={"modelID": str})

print(f"RNA-seq matrix shape: {rna.shape}")
print(f"Doubling-time table shape: {dt.shape}")

# ------------------------------------------------------------
# --- Merge by modelID
# ------------------------------------------------------------
merged = pd.merge(dt, rna, on="modelID", how="inner")

print(f"Merged dataset shape: {merged.shape}")
print(f"Common models retained: {merged['modelID'].nunique()}")

# ------------------------------------------------------------
# --- Save final file
# ------------------------------------------------------------
merged.to_csv(merged_out, index=False, float_format="%.3f")
print(f"✅ Merged dataset saved to:\n{merged_out}")

# Optional: quick summary
print("\nSample preview:")
print(merged.head(3))
