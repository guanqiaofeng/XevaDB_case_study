import pandas as pd

# === 1. Read geneInfo ===
gene_info = pd.read_csv("../../data/rawdata/geneInfo.tab", sep="\t", header=None, names=["ensembl_id", "symbol", "biotype"])

# Clean potential version suffixes (e.g., ENSG00000123456.5 → ENSG00000123456)
gene_info["ensembl_id"] = gene_info["ensembl_id"].str.split(".").str[0]

# === 2. Keep only protein_coding ===
gene_info = gene_info[gene_info["biotype"] == "protein_coding"]

# === 3. Drop symbols that map to multiple Ensembl IDs ===
# Count number of Ensembl IDs per symbol
symbol_counts = gene_info["symbol"].value_counts()
multi_symbol = symbol_counts[symbol_counts > 1].index
print(f"symbols that map to multiple Ensembl IDs: {multi_symbol.tolist()}")
gene_info = gene_info[~gene_info["symbol"].isin(multi_symbol)]

# === 4. Drop duplicated Ensembl IDs if any ===
# gene_info = gene_info.drop_duplicates(subset="ensembl_id", keep="first")

# === 5. Read TPM matrix ===
tpm = pd.read_csv("../../data/procdata/PDX_tpm_batchcorrected_2025Oct_common_model.csv", index_col=0)

# Remove version suffixes in TPM index as well
tpm.index = tpm.index.str.split(".").str[0]

# === 6. Filter TPM matrix for valid Ensembl IDs ===
valid_ensembl = set(gene_info["ensembl_id"])
tpm_filtered = tpm.loc[tpm.index.isin(valid_ensembl)]

# === 7. Map Ensembl ID → Hugo symbol ===
id_to_symbol = dict(zip(gene_info["ensembl_id"], gene_info["symbol"]))
tpm_filtered.index = tpm_filtered.index.map(id_to_symbol)

# === 8. Drop any genes without mapping (if any) ===
tpm_filtered = tpm_filtered[~tpm_filtered.index.isna()]
print(f"genes without mapping: {tpm_filtered[tpm_filtered.index.isna()].index.tolist()}")

# === 9. Drop any duplicate symbols just in case ===
tpm_filtered = tpm_filtered[~tpm_filtered.index.duplicated(keep="first")]

# === 10. Save cleaned file ===
tpm_filtered.to_csv("../../data/procdata/RNAseq_tpm_protein_coding_hugo.csv")

print(f"Final TPM shape: {tpm_filtered.shape}")
