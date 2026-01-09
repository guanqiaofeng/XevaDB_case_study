import pandas as pd

df = pd.read_excel("../../data/raw/sample_info.xlsx", sheet_name="Sheet2")

commom = pd.read_excel("../../data/process/commom_model_RNAseq_experiment.xlsx")

df = commom.merge(df, on="sample_name_standard", how="left")

df.to_excel("../../data/clean/common_model_RNAseq_info.xlsx", index=False)
