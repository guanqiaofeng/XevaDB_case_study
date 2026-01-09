import pandas as pd

df = pd.read_csv("../../data/raw/Cescon_PDX_tpm_batchcorrected_2025Oct.csv")

model_list = pd.read_excel("../../data/clean/common_model_RNAseq_info.xlsx")

# my df is a matrix with genes in rows and samples in columns. I want to extract the columns corresponding to the model_list.
# First, let's check what sample name column exists in model_list
sample_name_col = "sample_name" if "sample_name" in model_list.columns else "sample_name_standard"
# Keep the first column (gene ID) and add the sample columns
gene_col = df.columns[0]
selected_cols = [gene_col] + list(model_list[sample_name_col])
df = df.loc[:, selected_cols]

# Create a mapping dictionary for renaming columns
rename_dict = dict(zip(model_list[sample_name_col], model_list["sample_name_standard"]))
df = df.rename(columns=rename_dict)


df.to_csv("../../data/process/PDX_tpm_batchcorrected_2025Oct_common_model.csv", index=False)
