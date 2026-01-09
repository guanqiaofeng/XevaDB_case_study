#!/usr/bin/env python3
"""
14_SHAP_enrichment.py

Perform GO and KEGG enrichment on top SHAP-ranked genes
from XGBoost and LightGBM models predicting PDX doubling time.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gseapy import enrichr
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ------------------------------------------------------------
# --- Paths & Parameters
# ------------------------------------------------------------
ML_DIR = "../../data/ML"
OUT_PDF = os.path.join(ML_DIR, "SHAP_enrichment_report.pdf")
OUT_SUMMARY = os.path.join(ML_DIR, "SHAP_enrichment_summary.csv")

TOP_N_GENES = 100
TOP_TERMS = 15
GENE_SETS = ["GO_Biological_Process_2023", "KEGG_2021_Human"]

shap_files = {
    "LightGBM": os.path.join(ML_DIR, "LightGBM_SHAP_topgenes.csv"),
    "XGBoost": os.path.join(ML_DIR, "XGBoost_SHAP_topgenes.csv"),
}

# ------------------------------------------------------------
# --- Helper: run enrichment and plot
# ------------------------------------------------------------
def run_enrichment(gene_list, gene_set, model_name):
    """Run Enrichr enrichment using GSEApy"""
    res = enrichr(gene_list=gene_list, gene_sets=[gene_set], organism='Human', outdir=None, cutoff=0.5)
    df = res.results.sort_values("Adjusted P-value").head(TOP_TERMS)
    df["Model"] = model_name
    df["GeneSet"] = gene_set
    return df

def plot_enrichment(df, model_name, gene_set, pdf):
    """Generate horizontal barplot"""
    plt.figure(figsize=(7, 4))
    sns.barplot(
        x=-df["log10(p-value)"],
        y=df["Term"],
        color="steelblue"
    )
    plt.xlabel("-log10(p-value)")
    plt.ylabel("")
    plt.title(f"{model_name} — {gene_set} (top {TOP_TERMS})")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# ------------------------------------------------------------
# --- Main analysis
# ------------------------------------------------------------
all_results = []

with PdfPages(OUT_PDF) as pdf:
    for model_name, shap_path in shap_files.items():
        if not os.path.exists(shap_path):
            print(f"⚠️ {shap_path} not found — skipping {model_name}")
            continue

        print(f"\n🔍 Running enrichment for {model_name} ...")
        shap_df = pd.read_csv(shap_path)
        top_genes = shap_df["Gene"].head(TOP_N_GENES).tolist()

        for gene_set in GENE_SETS:
            try:
                df = run_enrichment(top_genes, gene_set, model_name)
                all_results.append(df)

                df_plot = df.copy()
                df_plot["log10(p-value)"] = -df_plot["Adjusted P-value"].apply(lambda x: 0 if x <= 0 else np.log10(x))
                plot_enrichment(df_plot, model_name, gene_set, pdf)

            except Exception as e:
                print(f"⚠️ Enrichment failed for {model_name} - {gene_set}: {e}")

# ------------------------------------------------------------
# --- Save combined results
# ------------------------------------------------------------
if all_results:
    res_all = pd.concat(all_results, ignore_index=True)
    res_all.to_csv(OUT_SUMMARY, index=False)
    print(f"\n✅ Saved enrichment summary → {OUT_SUMMARY}")
    print(f"📊 PDF report → {OUT_PDF}")
else:
    print("⚠️ No enrichment results generated.")
