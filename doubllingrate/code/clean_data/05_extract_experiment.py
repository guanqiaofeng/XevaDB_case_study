import pandas as pd

model_list = pd.read_csv("../../data/process/commom_model_RNAseq_experiment.txt", header=None)[0].tolist()

df = pd.read_csv("../../data/process/experiment_parental_H20.csv")

df = df[df["modelID"].isin(model_list)]

df.to_csv("../../data/clean/experiment_clean.csv", index=False)

