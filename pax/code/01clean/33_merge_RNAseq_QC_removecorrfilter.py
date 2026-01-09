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
rna_in = "../../data/procdata/RNAseq_log2tpm_filtered_for_ML_rmcorrfilter.csv"
dt_in  = "../../data/rawdata/PDX_BR_BAR_grouped_PAX.xlsx"
out_dir = "../../data/procdata"
os.makedirs(out_dir, exist_ok=True)
merged_out = os.path.join(out_dir, "RNAseq_with_pax_800gene.csv")

# ------------------------------------------------------------
# --- Load data
# ------------------------------------------------------------
rna = pd.read_csv(rna_in, dtype={"modelID": str})
dt  = pd.read_excel(dt_in, dtype={"PDX_ID": str}, usecols=["PDX_ID","mRECIST"])
dt = dt.rename(columns={"PDX_ID": "modelID", "mRECIST": "pax_mRECIST"})
dt["pax_cat"] = dt["pax_mRECIST"].isin(["CR", "PR"]).astype(int)
ord_map = {"PD": 0, "SD": 1, "PR": 2, "CR": 3}
dt["pax_ord"] = dt["pax_mRECIST"].map(ord_map)

valid = {"CR", "PR", "SD", "PD"}
bad = set(dt["pax_mRECIST"].dropna().unique()) - valid
if bad:
    raise ValueError(f"Unexpected mRECIST values: {bad}")

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

# Critical check #1: duplicates per modelID
assert merged["modelID"].is_unique, "Duplicate modelID detected!"

# Critical check #2: class balance after merge
print(merged["pax_cat"].value_counts())
print(merged["pax_mRECIST"].value_counts())