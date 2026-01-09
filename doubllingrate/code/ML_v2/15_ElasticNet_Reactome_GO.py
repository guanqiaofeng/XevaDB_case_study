import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Load coefficients
# ------------------------------------------------------------
df = pd.read_csv("../../data/ML_v2/14_ElasticNet_500HVG/ElasticNet_coefficients_500HVG.csv", index_col=0, header=None)
df.columns = ["coef"]

# Option 1: use all non-zero genes
gene_list = df.index[df["coef"] != 0].tolist()

# Option 2: OR use top 50 absolute coefficients (recommended)
# gene_list = df["coef"].abs().sort_values(ascending=False).head(50).index.tolist()

print(f"Using {len(gene_list)} genes for enrichment.")

# ------------------------------------------------------------
# Run Enrichr (GO BP + Reactome)
# ------------------------------------------------------------
libraries = [
    "GO_Biological_Process_2023",
    "Reactome_2022"
]

results = {}

for lib in libraries:
    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=[lib],
        organism="Human",
        outdir=None,        # no folder creation
        cutoff=1.0          # include all results
    )
    results[lib] = enr.results

# ------------------------------------------------------------
# Save CSV results
# ------------------------------------------------------------
for lib, table in results.items():
    outname = f"ElasticNet_500HVG_{lib}_enrichment.csv"
    table.to_csv(outname, index=False)
    print(f"Saved → {outname}")

# ------------------------------------------------------------
# Optional barplots
# ------------------------------------------------------------
for lib, table in results.items():
    top10 = table.head(20)

    plt.figure(figsize=(14,10))
    sns.barplot(
        data=top10,
        x="Combined Score",
        y="Term",
        color="teal"
    )
    plt.title(f"Top 10 Enriched Terms ({lib})")
    plt.tight_layout()
    plt.savefig(f"{lib}_top10_barplot.png", dpi=150)
    plt.close()

    print(f"Saved → {lib}_top10_barplot.png")
