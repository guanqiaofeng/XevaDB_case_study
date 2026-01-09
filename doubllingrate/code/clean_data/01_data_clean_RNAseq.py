import pandas as pd
import numpy as np

df = pd.read_excel("../../data/raw/sample_info.xlsx")

model_list_RNAseq = sorted(df["sample_name_standard"].dropna().unique().astype(str))
#pd.Series(model_list, name="modelID").to_csv("../../data/process/RNAseq_models.csv", index=False)


pd.Series(model_list_RNAseq, name="sample_name_standard").to_csv("../../data/process/RNAseq_models.txt", index=False)


# #I want to generate 3 list, one shared by model_list_RNAseq and model_list_experiment, one only in model_list_RNAseq, one only in model_list_experiment.
# model_list_shared = [model for model in model_list_RNAseq if model in model_list_experiment]
# model_list_RNAseq_only = [model for model in model_list_RNAseq if model not in model_list_experiment]
# model_list_experiment_only = [model for model in model_list_experiment if model not in model_list_RNAseq]

# pd.Series(model_list_shared, name="modelID").to_csv("../../data/process/RNAseq_models_shared.csv", index=False)
# pd.Series(model_list_RNAseq_only, name="modelID").to_csv("../../data/process/RNAseq_models_only.csv", index=False)
# pd.Series(model_list_experiment_only, name="modelID").to_csv("../../data/process/experiment_models_only.csv", index=False)